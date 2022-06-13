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

"""Input data for image models.

For performance reasons the format used is NCHW.
"""

import functools
import glob
import itertools
import os

import numpy as np
import tensorflow as tf

from libml import utils

# _DATA_CACHE = None
# DATA_DIR = os.environ['ML_DATA']

class LagDataLoader:
    def __init__(self, data_name, data_dir, data_size, channel):
        self.data_name = data_name
        self.data_dir = data_dir
        self.data_size = data_size
        self.channel = channel
        self.colors = channel
        self.height = data_size
        self.width = data_size


        dataSetName_train = self.data_name + '-train.tfrecord'
        dataSetName_test = self.data_name + '-test.tfrecord'
        self.train = self.parse_tfrecord([os.path.join(data_dir, dataSetName_train)])
        self.test = self.parse_tfrecord([os.path.join(data_dir, dataSetName_test)])

        self.graph = None  # TensorFlow graph
        self.sess = None  # TensorFlow session

    @tf.function
    def parse_tfrecord(self, filenames, para=4):
        """
        Operations:
            load tfrecord
            decode float32 image data, then scale from [0,255] to [-1,1]
            crop / resize / flip / transpose
            finally get HWC images and corresponding labels
        :Args:
            filenames: must be a list!!!!!
        Return:
            dict(x=x, label=label) x:HWC
        """

        # the number elements to process asynchronously in parallel
        para *= 4 * max(1, len(utils.get_available_gpus()))
        # filenames: tfrecord format dataset file path
        filenames = sorted(sum([glob.glob(x) for x in filenames], []))
        if not filenames:
            raise ValueError('Empty dataset')
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.record_parse_fn, num_parallel_calls=para)

        return dataset.map(self.iterator_dict)

    @tf.function
    def record_parse_fn(self, record):
        """
        parse each element in the tfrecord dataset file

        decode:
            image(float), original_lowest(float), original_highest(float), shape(Int64), label(byte)

        scale image from [original_lowest,original_highest] to [-1,1] for training

        Return: image, label
        """
        image_feature_description = {
            'data': tf.io.VarLenFeature(tf.float32),
            'label': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'shape': tf.io.FixedLenFeature([3], tf.int64, default_value=[0, 0, 0]),
        }

        features = tf.io.parse_single_example(
            record, features=image_feature_description
        )

        image = tf.sparse_tensor_to_dense(features['data'], default_value=0.0)
        label = features['label']
        shape = features['shape']
        image = tf.reshape(image, shape) # HWC
        size = [self.data_size,self.data_size]
        if not size == shape[:-1]:
            image = tf.image.resize(
                    image,
                    size,
                    method=tf.image.ResizeMethod.BICUBIC
                )
        image = tf.transpose(image,perm=[2,0,1]) # CHW
        original_lowest = tf.math.reduce_min(image)
        original_highest = tf.math.reduce_max(image)

        # stretch from [original_lowest, original_highest] to [-1,1] for training
        image = tf.math.add(
            tf.math.divide(
                tf.math.multiply(tf.math.add(image, -original_lowest), 2.0),
                tf.math.maximum(tf.math.add(original_highest, -original_lowest), 1e-5),
            ),
            -1.0,
        )

        return image, label, shape, original_lowest, original_highest

    @staticmethod
    def iterator_dict(x, label, shape, original_lowest, original_highest):
        return dict(
            x=x,
            label=label,
            shape=shape,
            original_lowest=original_lowest,
            original_highest=original_highest,
        )


class DataSet:
    def __init__(self, name, train, test, height, width, colors, nclass, nattr=0):
        self.name = name
        self.train = train
        self.test = test
        self.height = height
        self.width = width
        self.colors = colors
        self.nclass = nclass
        self.nattr = nattr  # Attributes, number of float attributes.
        self.graph = None  # TensorFlow graph
        self.sess = None  # TensorFlow session

    @staticmethod
    def record_parse_fn(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={'image': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.int64)})
        image = tf.image.decode_image(features['image'])
        image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
        label = features['label']
        return image, label

    def to_record_iterator(self, filenames, size, resize, repeat=False, random_flip_x=False,
                           limit=0, crop=(0, 0), para=4):
        para *= 4 * max(1, len(utils.get_available_gpus()))
        filenames = sorted(sum([glob.glob(x) for x in filenames], []))
        if not filenames:
            raise ValueError('Empty dataset, did you mount gcsfuse bucket?')
        dataset = tf.data.TFRecordDataset(filenames)
        if limit is not None:
            if limit > 0:
                dataset = dataset.take(limit)
            elif limit < 0:
                dataset = dataset.skip(-limit)
        if repeat:
            dataset = dataset.repeat()

        def fpy(image):
            # Faster than using tf primitives.
            image = image.transpose(2, 0, 1)
            if random_flip_x and np.random.randint(2):
                image = image[:, :, ::-1]
            return image

        def f(image, *args):
            delta = [0, 0]
            if sum(crop):
                image = image[crop[0]:-crop[0], crop[1]:-crop[1]]
                delta[0] -= 2 * crop[0]
                delta[1] -= 2 * crop[1]
            if resize[0] - delta[0] != size[0] or resize[1] - delta[1] != size[1]:
                image = tf.image.resize_area([image], list(resize))[0]
            image = tf.py_func(fpy, [image], tf.float32)
            image = tf.reshape(image, [size[-1]] + list(resize))
            return (image,) + args

        dataset = dataset.map(self.record_parse_fn, num_parallel_calls=para)
        dataset = dataset.filter(lambda image, *_: tf.equal(tf.shape(image)[2], size[2]))
        dataset = dataset.map(f, para)
        return dataset.map(self.iterator_dict)

    @staticmethod
    def iterator_dict(x, label):
        return dict(x=x, label=label)


class DataSetSemi(DataSet):
    """Semi-supervised dataset: train has labels, unlabeled has no labels, test has labels."""

    def __init__(self, name, train, unlabeled, test, height, width, colors, nclass):
        DataSet.__init__(self, name, train, test, height, width, colors, nclass)
        self.unlabeled = unlabeled


class DataSetCelebA(DataSet):
    def __init__(self, height, width):
        shared_kwargs = dict(size=(218, 178, 3), crop=(30, 10), resize=(height, width))
        train = self.to_record_iterator([os.path.join(DATA_DIR, 'celeba-train.tfrecord')],
                                        repeat=True, random_flip_x=True, **shared_kwargs)
        train = train.shuffle(1024)
        test = self.to_record_iterator([os.path.join(DATA_DIR, 'celeba-test.tfrecord')],
                                       **shared_kwargs)
        DataSet.__init__(self, 'celeba%d' % width, train, test, height, width, 3, nclass=10177, nattr=40)

    @staticmethod
    def record_parse_fn(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={'image': tf.FixedLenFeature([], tf.string),
                      'label': tf.FixedLenFeature([], tf.int64),
                      'attrs': tf.FixedLenFeature([40], tf.float32)})
        image = tf.image.decode_image(features['image'])
        image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
        return image, features['label'], features['attrs']

    @staticmethod
    def iterator_dict(x, label, attrs):
        return dict(x=x, label=label, attrs=attrs)


class DataSetCifar10(DataSet):
    def __init__(self):
        shared_kwargs = dict(size=(32, 32, 3), resize=(32, 32))
        train = self.to_record_iterator([os.path.join(DATA_DIR, 'cifar10-train.tfrecord')],
                                        repeat=True, random_flip_x=True, **shared_kwargs)
        train = train.shuffle(1024)
        test = self.to_record_iterator([os.path.join(DATA_DIR, 'cifar10-test.tfrecord')],
                                       **shared_kwargs)
        DataSet.__init__(self, 'cifar10', train, test, 32, 32, 3, nclass=10)


class DataSetMNIST(DataSet):
    def __init__(self):
        shared_kwargs = dict(size=(28, 28, 1), resize=(28, 28))
        train = self.to_record_iterator([os.path.join(DATA_DIR, 'mnist-train.tfrecord')],
                                        repeat=True, **shared_kwargs)
        train = train.shuffle(1024)
        test = self.to_record_iterator([os.path.join(DATA_DIR, 'mnist-test.tfrecord')],
                                       **shared_kwargs)
        DataSet.__init__(self, 'mnist', train, test, 28, 28, 1, nclass=10)


class DataSetSVHN(DataSet):
    def __init__(self):
        shared_kwargs = dict(size=(32, 32, 3), resize=(32, 32))
        train = self.to_record_iterator([os.path.join(DATA_DIR, 'svhn-train.tfrecord'),
                                         os.path.join(DATA_DIR, 'svhn-extra.tfrecord')],
                                        repeat=True, **shared_kwargs)
        train = train.shuffle(1024)
        test = self.to_record_iterator([os.path.join(DATA_DIR, 'svhn-test.tfrecord')],
                                       **shared_kwargs)
        DataSet.__init__(self, 'svhn', train, test, 32, 32, 3, nclass=10)


class DataSetLSUN(DataSet):
    def __init__(self, category, height, width):
        shared_kwargs = dict(size=(height + 1, width + 1, 3), resize=(height, width))
        train = self.to_record_iterator([os.path.join(DATA_DIR, 'lsun_%s-train.tfrecord' % category)],
                                        repeat=True, random_flip_x=True, **shared_kwargs)
        train = train.shuffle(1024)
        test = self.to_record_iterator([os.path.join(DATA_DIR, 'lsun_%s-val.tfrecord' % category)],
                                       **shared_kwargs)
        DataSet.__init__(self, '%s%d' % (category, width), train, test, height, width, 3, nclass=0)


class DataSetImageNet(DataSet):
    def __init__(self, height, width):
        shared_kwargs = dict(size=(height + 1, width + 1, 3), resize=(height, width))
        train = self.to_record_iterator([os.path.join(DATA_DIR, 'imagenet-2012-tfrecord/train-*-of-01024')],
                                        repeat=True, random_flip_x=True, **shared_kwargs)
        train = train.shuffle(1024)
        test = self.to_record_iterator([os.path.join(DATA_DIR, 'imagenet-2012-tfrecord/validation-*-of-00128')],
                                       **shared_kwargs)
        DataSet.__init__(self, 'imagenet%d' % width, train, test, height, width, 3, nclass=1000)

    @staticmethod
    def record_parse_fn(serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={'image/encoded': tf.FixedLenFeature([], tf.string),
                      'image/class/label': tf.FixedLenFeature([], tf.int64)})
        image = tf.image.decode_image(features['image/encoded'])
        image = tf.cast(image, tf.float32) * (2.0 / 255) - 1.0
        label = features['image/class/label'] - 1
        return image, label


class DataSetAll(DataSet):
    """A dataset that interleaves all datasets of a given size (except imagenet)."""

    def __init__(self, width):
        datasets = [y for x, y in _DATASETS.items()
                    if x.endswith(str(width)) and x != 'all%d' % width and 'imagenet' not in x]
        datasets = [x() for x in datasets]

        def flatten_and_label(*xs):
            tx = [tf.data.Dataset.from_tensors(x).map(lambda x: dict(x=x, label=tf.constant(p)))
                  for p, x in enumerate(xs)]
            t = tx.pop()
            while tx:
                t = t.concatenate(tx.pop())
            return t

        train = [x.train.map(lambda x: x['x']) for x in datasets]
        train = tf.data.Dataset.zip(tuple(train))
        train = train.flat_map(flatten_and_label)
        test = [x.test.map(lambda x: x['x']) for x in datasets]
        test = tf.data.Dataset.zip(tuple(test))
        test = test.flat_map(flatten_and_label)
        DataSet.__init__(self, 'all%d' % width, train, test, width, width, 3, len(datasets))


def as_iterator(data, sess):
    # it = data.make_one_shot_iterator().get_next()
    it = tf.compat.v1.data.make_one_shot_iterator(data).get_next()

    def iterator():
        while 1:
            yield sess.run(it)

    return iterator()


# def get_dataset(dataset_name):
def get_dataset(dataset_name, data_dir, data_size, channel):
    g = tf.Graph()
    with g.as_default():
        # dataset = _DATASETS[dataset_name]()
        dataset = LagDataLoader(dataset_name, data_dir, data_size, channel)
        dataset.graph = g
        dataset.sess = tf.Session()
        return dataset


_LSUN_CATEGORIES = ['bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room',
                    'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower']
_LSUN = {'lsun_%s%d' % (category, size): functools.partial(DataSetLSUN, category, size, size)
         for category, size in itertools.product(_LSUN_CATEGORIES, [32, 64, 128, 256])}

_DATASETS = {
    'all32': functools.partial(DataSetAll, 32),
    'all64': functools.partial(DataSetAll, 64),
    'all128': functools.partial(DataSetAll, 128),
    'all256': functools.partial(DataSetAll, 256),
    'celeba32': functools.partial(DataSetCelebA, 32, 32),
    'celeba64': functools.partial(DataSetCelebA, 64, 64),
    'celeba128': functools.partial(DataSetCelebA, 128, 128),
    'celeba256': functools.partial(DataSetCelebA, 256, 256),
    'cifar10': DataSetCifar10,
    'imagenet32': functools.partial(DataSetImageNet, 32, 32),
    'imagenet64': functools.partial(DataSetImageNet, 64, 64),
    'imagenet128': functools.partial(DataSetImageNet, 128, 128),
    'imagenet256': functools.partial(DataSetImageNet, 256, 256),
    'mnist': DataSetMNIST,
    'svhn': DataSetSVHN,
    **_LSUN
}
