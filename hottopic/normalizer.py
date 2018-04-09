#normalizer.py

import os
import json
import numpy as np

import hottopic as ht

# class Normalizer2()

class Normalizer(object):

    def __init__(self, fits=None):
        self.fits = fits

    def fit(self, inputs):
        '''Find the mean and std dev of each channel for this group of inputs
        ignore the first channel, the starting_perim'''
        if not isinstance(inputs, list):
            inputs = [inputs]
        self.fits = []
        for branch in inputs:
            num_channels = branch[0].shape[-1]
            channel_fits = []
            for i in range(num_channels):
                channel = np.concatenate([inp[...,i].ravel() for inp in branch])
                mean, std = np.nanmean(channel), np.nanstd(channel)
                channel_fits.append((float(mean), float(std)))
            self.fits.append(channel_fits)

    def normalize(self, inputs):
        if self.fits is None:
            raise ValueError("Trying to normalize with an unfitted Normalizer")
        assert isinstance(inputs, list)
        if len(inputs) != len(self.fits):
            raise ValueError(("Fitted on inputs with {} branches"
            " but got input with {} branches").format(len(self.fits), len(inputs)))
        for branch, fits in zip(inputs, self.fits):
            if branch.shape[-1] != len(fits):
                raise ValueError(("Fitted on a branch with {} channels"
                " but got branch with {} channels").format(len(fits), branch.shape[-1]))
            for i, (mean, std) in enumerate(fits):
                channel = branch[...,i]
                channel = (channel-mean) / std
                branch[...,i] = channel
        return inputs

    def save(self, fname):
        if self.fits is None:
            raise ValueError("Saving a non-fitted Normalizer")
        if 'preprocessors' + os.sep not in fname:
            fname = 'preprocessors' + os.sep + fname
        if not fname.endswith('.json'):
            fname += '.json'
        with open(fname, 'w') as fp:
            json.dump(self.fits, fp, sort_keys=True, indent=4)

    @staticmethod
    def fromFile(fname):
        if 'preprocessors' + os.sep not in fname:
            fname = 'preprocessors' + os.sep + fname
        if not fname.endswith('.json'):
            fname += '.json'
        with open(fname, 'r') as fp:
            fits = json.load(fp)
        return Normalizer(fits=fits)

    def __repr__(self):
        return 'Normalizer({})'.format(self.fits)

def fromFile(fname):
    return Normalizer.fromFile(fname)
