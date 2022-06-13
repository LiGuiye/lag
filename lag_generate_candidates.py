#!/usr/bin/env python

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to download all datasets and create .tfrecord files.
"""
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from easydict import EasyDict
from tensorflow.python.framework import dtypes
from tqdm import trange

from lag import LAG
from libml import data, utils

FLAGS = flags.FLAGS


def get_ops(dataset):
    ckpt = LAG.load(FLAGS.ckpt).eval_mode(dataset=dataset)

    def eval(batch):
        v = ckpt.sess.run(ckpt.ops.eval_op, feed_dict={ckpt.ops.x: batch})
        return v

    def sres(batch, noise):
        v = ckpt.sess.run(ckpt.ops.sres_op, feed_dict={ckpt.ops.y: batch, ckpt.ops.noise: noise})
        return v

    def lores(batch):
        return ckpt.sess.run(ckpt.ops.downscale_op, feed_dict={ckpt.ops.x: batch})

    def hires(batch):
        return ckpt.sess.run(ckpt.ops.upscale_op, feed_dict={ckpt.ops.y: batch})

    return EasyDict(eval=eval, sres=sres, lores=lores, hires=hires)


def get_samples_indexes(txt_list):
    l = []
    for v in txt_list:
        try:
            if '-' in v:
                l.extend(list(range(int(v.split('-')[0]), int(v.split('-')[1]) + 1)))
            else:
                l.append(int(v))
        except:
            l.append(int(v))
    return sorted(set(l))


def get_candidates(ops, images):
    batch = FLAGS.batch
    lores = np.concatenate([ops.lores(images[x:x + batch]) for x in range(0, images.shape[0], batch)], axis=0)
    hires = np.concatenate([ops.hires(lores[x:x + batch]) for x in range(0, images.shape[0], batch)], axis=0)
    zoomed = np.stack([np.concatenate([ops.sres(lores[x:x + batch], y / (FLAGS.ncand - 1))
                                       for x in range(0, images.shape[0], batch)], axis=0)
                       for y in trange(FLAGS.ncand, leave=False, desc='Generate')], axis=1)
    return np.concatenate([images[:, None], hires[:, None], zoomed], axis=1)


def load_hires(dataset, indexes):
    indexes = frozenset(indexes)

    def match(k):
        return k in indexes

    with dataset.graph.as_default():
        max_value = np.iinfo(dtypes.int64.as_numpy_dtype).max
        test_data = tf.data.Dataset.zip((tf.data.Dataset.range(max_value), dataset.test))
        test_data = test_data.filter(lambda k, v: tf.py_func(match, [k], tf.bool)).map(lambda k, v: v['x'])
        test_data = test_data.batch(len(indexes)).make_one_shot_iterator().get_next()
        with tf.Session() as sess_data:
            return sess_data.run(test_data)


def main(args):
    del args
    dataset_name = FLAGS.dataset or os.path.basename(os.path.dirname(os.path.dirname(FLAGS.ckpt)))
    try:
        # dataset = data.get_dataset(dataset_name)
        if FLAGS.dataset == 'Wind_2007':
            root = '/home/guiyli/Documents/DataSet/Wind/2007/u_v'
            # root = '/home/guiyli/DataSet/Wind/2007/u_v'
        if FLAGS.dataset == 'Solar_2009':
            root = '/home/guiyli/Documents/DataSet/NSRDB/500X500/2009/grid1/dni_dhi'
            # root = '/home/guiyli/DataSet/Solar/2009/dni_dhi'
        dataset = data.get_dataset(dataset_name=FLAGS.dataset, data_dir=root, data_size=256, channel=2)
    except KeyError:
        dataset = data.get_dataset('lsun_' + dataset_name)
    ops = get_ops(dataset)
    images = load_hires(dataset, get_samples_indexes(FLAGS.samples))
    image_grid = get_candidates(ops, images)
    img = utils.images_to_grid(image_grid) # HWC
    # output_file = os.path.abspath(FLAGS.save_to)
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # open(output_file, 'wb').write(utils.to_png(img))
    # print('Saved', output_file)
    _, _, c = img.shape
    for i in range(c):
        output_file = os.path.abspath('test'+str(i)+'.png')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        open(output_file, 'wb').write(utils.to_png(img[:,:,i][:, :, np.newaxis]))
        print('Saved', output_file)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_string('save_to', 'test.png', 'Path were to save candidates.')
    # flags.DEFINE_string('ckpt', '', 'Path where to load the trained lag model.')
    flags.DEFINE_string('ckpt', 'experiments/Solar_2009/average32X/LAG_batch4_blocks8_filters256_filters_min64_lod_min1_lr0.001_mse_weight10.0_noise_dim64_training_kimg118_transition_kimg118_ttur4_wass_target1.0_weight_avg0.999', 'Path where to load the trained lag model.')
    flags.DEFINE_list('samples', [0,3], 'Index of samples to retrieve.')
    flags.DEFINE_integer('ncand', 16, 'Number of candidates to generate per image.')
    # FLAGS.set_default('dataset', '')  # To override model dataset.
    # FLAGS.set_default('batch', 64)
    app.run(main)
