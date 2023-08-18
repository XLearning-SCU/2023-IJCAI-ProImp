import os
import random
import sys
import numpy as np
import scipy.io as sio
from scipy import sparse
import torch
import torchvision


def load_data(config):
    """Load data """
    data_name = config['dataset']
    main_dir = sys.path[0]
    X_list = []
    Y_list = []

    if data_name in ['Scene_15']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Scene-15.mat'))
        X = mat['X'][0]
        X_list.append(X[0].astype('float32'))
        X_list.append(X[1].astype('float32'))
        Y_list.append(np.squeeze(mat['Y']))


    elif data_name in ['NoisyMNIST']:
        mat = sio.loadmat(os.path.join(main_dir, 'data','NoisyMNIST30000.mat'))
        X_list.append(mat['X1'])
        X_list.append(mat['X2'])
        Y_list.append(np.squeeze(mat['Y']))


    elif data_name in ['Reuters']:
        mat = sio.loadmat(os.path.join(main_dir, 'data','Reuters.mat'))
        X_list.append(
            normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        X_list.append(
            normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        Y_list.append(np.squeeze(np.hstack((mat['y_train'], mat['y_test']))))


    elif data_name in ['MNIST-USPS']:
        mat = sio.loadmat(os.path.join(main_dir, 'data','MNIST-USPS.mat'))
        X_list.append(mat['X1'])
        X_list.append(normalize(mat['X2']))
        Y_list.append(np.squeeze(mat['Y']))


    elif data_name in ['cub_googlenet']:
        mat = sio.loadmat(os.path.join(main_dir, 'data','cub_googlenet_doc2vec_c10.mat'))
        X_list.append(normalize(mat['X'][0][0]))
        X_list.append(normalize(mat['X'][0][1]))
        Y_list.append(np.squeeze(mat['gt']))


    return X_list, Y_list


def trans(data, normalize_type='dim_wise', withTanh=0, DimensionalityReduction=0):

    new_xs = []
    for x in data:
        x = torch.from_numpy(x)
        x = x.view((-1, 1, 1, x.shape[-1]))
        if normalize_type == 'dim_wise':
            mean, std = torch.mean(x, dim=0), torch.std(x, dim=0)
            std[std < torch.max(std) * 1e-6] = 1
            x = torchvision.transforms.Normalize(mean, std)(x)
        elif normalize_type == 'sample_wise':
            mean, std = torch.mean(x), torch.std(x)
            x = torchvision.transforms.Normalize(mean, std)(x)
        elif normalize_type == 'rescale_dim_wise':
            ma = torch.amax(x, dim=0)
            mi = torch.amin(x, dim=0)
            mean, std = (ma + mi) / 2, (ma - mi) / 2
            std[std < torch.max(std) * 1e-6] = 1
            x = torchvision.transforms.Normalize(mean, std)(x)
        elif normalize_type == 'rescale_sample_wise':
            ma = torch.amax(x)
            mi = torch.amin(x)
            mean, std = (ma + mi) / 2, (ma - mi) / 2
            x = torchvision.transforms.Normalize(mean, std)(x)
        elif normalize_type == 'None':
            pass
        else:
            raise NotImplementedError("")
        if withTanh:
            x = torch.nn.Tanh()(x)
        x = x.view((-1, x.shape[-1])).numpy()
        if DimensionalityReduction:
            if x.shape[1] != DimensionalityReduction:
                x = PCA(n_components=DimensionalityReduction).fit_transform(x)
        new_xs.append(x)
    return new_xs


def normalize(x):
    x = (x-np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile(
        (np.max(x, axis=0)-np.min(x, axis=0)), (x.shape[0], 1))
    return x


class DataSet_NoisyMNIST(object):

    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (np.uint8, np.float32):
            raise TypeError(
                'Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                        labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                        labels.shape))
            self._num_examples = images1.shape[0]

            if dtype == np.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                print("type conversion view 1")
                images1 = images1.astype(np.float32)

            if dtype == np.float32 and images2.dtype != np.float32:
                print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_image for _ in range(batch_size)], [fake_label for _
                                                                                                      in range(
                                                                                                          batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]
