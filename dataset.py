from __future__ import division

import pickle as pkl
import math
import numpy as np


class Dataset(object):

    def __init__(self, is_binary=False):
        self.is_binary = is_binary

        #Examples
        self.Xtrain = None
        self.Xtest = None

        #Labels
        self.Ytrain = None
        self.Ytest = None

        self.Xtrain_pres = None
        self.Xtest_pres = None

        self.sparsity = 0.0
        self.n_examples = 0

    def _get_data(self, data_path):
        if data_path.endswith("pkl") or data_path.endswith("pickle"):
            data = pkl.load(open(data_path, "rb"))
        else:
            data = np.load(data_path)
        return data

    def binarize_labels(self, labels=None):
        #Largest label is for the images without different object.
        last_lbl = np.max(labels)
        binarized_lbls = []
        if self.is_binary:
            for label in labels:
                if label == last_lbl:
                    binarized_lbls.append(0)
                else:
                    binarized_lbls.append(1)
        return binarized_lbls

    def setup_dataset(self, data_path=None, train_split_scale = 0.8):

        data = self._get_data(data_path)
        self.n_examples = data[0].shape[0]
        ntrain = math.floor(self.n_examples * train_split_scale)

        self.Xtrain = data[0][:ntrain]
        self.Xtrain_pres = data[2][:ntrain]
        self.Xtest = data[0][ntrain:]
        self.Xtest_pre = data[2][ntrain:]

        if train_split_scale != 0.0:
            self.Ytrain = np.array(self.binarize_labels(data[1][:ntrain].flatten()) \
            if self.is_binary else data[1][:ntrain].flatten())

        if train_split_scale != 1.0:
            self.Ytest = np.array(self.binarize_labels(data[1][ntrain:].flatten()) \
            if self.is_binary else data[1][ntrain:].flatten())

    def comp_sparsity(self):
        num_sparse_els = 0
        for el in self.Xtrain.flatten():
            if el == 0:
                num_sparse_els+=1
        for el in self.Xtest.flatten():
            if el == 0:
                num_sparse_els+=1
        self.sparsity = (num_sparse_els/self.n_examples)
        return self.sparsity
