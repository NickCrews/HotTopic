#normalizer.py

import os
import json
import numpy as np

import hottopic as ht

class Normalizer(object):

    def __init__(self, fits=None):
        self.fits = fits

    def fit(self, inputs):
        '''Find the mean and std dev of each channel for this group of inputs
        ignore the first channel, the starting_perim'''
        self.num_channels = inputs[0].shape[-1]
        self.fits = []
        for i in range(1, self.num_channels):
            channel = [inp[:,:,i] for inp in inputs]
            s = sum(np.nansum(c) for c in channel)
            N = sum(c.size for c in channel)
            mean =  s/N
            devs_squared = [(c-mean)**2 for c in channel]
            sum_devs_squared = sum(np.nansum(d) for d in devs_squared)
            std = np.sqrt(sum_devs_squared/N)
            print(mean, std)
            self.fits.append((mean, std))

    def normalize(self, inputs):
        if self.fits is None:
            raise ValueError("Trying to normalize with an unfitted Normalizer")
        if isinstance(inputs, np.ndarray):
            return self._normalize(inputs)
        else:
            return [self._normalize(inp) for inp in inputs]

    def _normalize(self, input):
        if input.shape[-1] != self.num_channels:
            raise ValueError("Fitted with inputs that have a different number of channels")

        for i, (mean, std) in enumerate(self.fits, start=1):
            # ignore the first channel, starting_perim
            channel = inp[:,:,i]
            channel = (channel-mean) / std
        return input

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
        return PreProcessor(fits=fits)

    def __repr__(self):
        return 'Normalizer({})'.format(self.fits)
