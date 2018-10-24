import sys
sys.path.append('/home/john/Desktop/caffe-master/python/')

import caffe

import numpy as np
from PIL import Image
import scipy.io

import random

class LoadDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - nyud_dir: path to dir
        - split: train / val / test
        - tops: list of tops to output
        - randomize: load in random order
        - seed: seed for randomization (default: None / current time)

        example: params = dict(nyud_dir="/path/to/STARE", split="val",
                               tops=['color', 'label'])
        """
        # config
        params = eval(self.param_str)
        self.nyud_dir = params['nyud_dir']
        self.split = params['split']
        self.tops = params['tops']
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # store top data for reshape + forward
        self.data = {}

        # means
        self.mean_bgr = np.array((126.8371, 69.0155, 41.4216), dtype=np.float32)
        self.mean_d = np.array((98.1881), dtype=np.float32)

        # tops: check configuration
        if len(top) != len(self.tops):
            raise Exception("Need to define {} tops for all outputs.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.nyud_dir, self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        # load data for tops and  reshape tops to fit (1 is the batch dim)
        for i, t in enumerate(self.tops):
            self.data[t] = self.load(t, self.indices[self.idx])
            top[i].reshape(1, *self.data[t].shape)

    def forward(self, bottom, top):
        # assign output
        for i, t in enumerate(self.tops):
            top[i].data[...] = self.data[t]

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load(self, top, idx):
        if top == 'color':
            return self.load_image(idx)
        elif top == 'label':
            return self.load_label(idx)
        elif top == 'IDMask':
            return self.load_IDMask(idx)
        elif top == 'Thickness':
            return self.load_Thickness(idx)
        else:
            raise Exception("Unknown output type: {}".format(top))

    def load_IDMask(self, idx):
        """
        Load skeleton (range) map as 1 x height x width integer array of skeleton segments.
        """
        IDMask = scipy.io.loadmat('{}/Range/{}.mat'.format(self.nyud_dir, idx))['IDMask'].astype(np.float32)
        IDMask = IDMask[np.newaxis, ...]
        return IDMask

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/Image/{}.png'.format(self.nyud_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ -= self.mean_d
        in_ = in_[np.newaxis, ...]

        return in_

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        Shift labels so that classes are 0-1 and void is 255 (to ignore it).
        The leading singleton dimension is required by the loss.
        """
        label = scipy.io.loadmat('{}/Mat/{}.mat'.format(self.nyud_dir, idx))['label'].astype(np.int16)
        label = label[np.newaxis, ...]
        return label
