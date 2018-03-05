
import random

import numpy as np
import cv2

import hottopic as ht

VULNERABLE_RADIUS = 750 #in meters
BATCH_SIZE = 32 #number of samples per batch of training

class Sample(object):

    stackedAndPadded = {}

    def __init__(self, spec, day, loc):
        self.spec = spec
        self.day = day
        self.loc = loc

    def getSpatialData(self):
        if (self.spec, self.day) in Sample.stackedAndPadded:
            sp = Sample.stackedAndPadded[(self.spec, self.day)]
        else:
            sp = self.stackAndPad()
            Sample.stackedAndPadded[(self.spec, self.day)] = sp
        y,x = self.loc
        r = self.spec.AOIRadius
        #add r to everything to deal with indexing change of padding
        loy, hiy, lox, hix = y-r+r, y+r+1+r, x-r+r, x+r+1+r
        result = sp[loy:hiy, lox:hix]
        return result

    def getNonSpatialData(self):
        rw = self.day.weather #rawWeather
        precip = ht.util.totalPrecipitation(rw)
        temp = ht.util.maximumTemperature1(rw)
        temp2 = ht.util.maximumTemperature2(rw)
        hum = ht.util.averageHumidity(rw)
        winds = ht.util.windMetrics(rw)
        allMetrics = [precip, temp, temp2, hum] + winds
        return np.array(allMetrics)

    def getOutput(self):
        return self.day.endingPerim[self.loc]

    def stackAndPad(self):
        layers = [self.day.startingPerim] + [self.day.burn.layers[name] for name in self.spec.layerNames]
        stacked = np.dstack(layers)
        r = self.spec.AOIRadius
        # pad with zeros around border of image
        padded = np.lib.pad(stacked, ((r,r),(r,r),(0,0)), 'constant')
        return padded

    def __repr__(self):
        return 'Sample({}, {})'.format(self.day, self.loc)

class SampleSpec(object):
    AOIRadius = 30
    AOISize = (2*AOIRadius+1, 2*AOIRadius+1)
    layerNames = ['dem', 'ndvi']
    numLayers = len(layerNames)
    nonLayers = ['rel_hum', 'temp', 'dew_pt']
    numNonLayers = len(nonLayers)

def makeSamples(days, spec=SampleSpec):
    samples = []
    for day in days:
        vulnerable = vulnerablePixels(day)
        evened = evenOutPositiveAndNegative(day, vulnerable)
        ys, xs =  np.where(evened)
        print('Making samples from {} locations in the day {}'.format(len(xs), day))
        for y,x in zip(ys,xs):
            s = Sample(spec, day, (y,x))
            samples.append(s)
    return samples

def getBatches(samples, batchSize=BATCH_SIZE, shuffle=True):
    if shuffle:
        random.shuffle(samples)
    for i in range(0, len(samples), batchSize):
        yield samples[i:i+batchSize]

def toModelInput(batch):
    spatials = np.array([sample.getSpatialData() for sample in batch])
    nonSpatials = np.array([sample.getNonSpatialData() for sample in batch])
    return [nonSpatials, spatials]

def toModelOutput(batch):
    return np.array([sample.getOutput() for sample in batch])

def vulnerablePixels(day, radius=VULNERABLE_RADIUS):
    '''Return a boolean mask of the pixels that are close to the current fire perimeter. radius is in meters.'''
    startingPerim = day.startingPerim.astype(np.uint8)
    kernel = np.ones((3,3))
    its = int(round((2*(radius/ht.rawdata.PIXEL_SIZE)**2)**.5))
    dilated = cv2.dilate(startingPerim, kernel, iterations=its)
    border = dilated - startingPerim
    return border.astype(np.uint8)

def evenOutPositiveAndNegative(day, mask):
    '''Make it so the chosen pixels for a day is a more even mixture of yes and no outputs.
    day is the Day to look at. mask is the boolean mask of locations we are looking at now'''
    didBurn = day.endingPerim.astype(np.uint8)
    didNotBurn = 1-didBurn
    # all the pixels we are training on that DID and did NOT burn
    pos = np.bitwise_and(didBurn, mask)
    neg = np.bitwise_and(didNotBurn, mask)
    numPos = np.count_nonzero(pos)
    numNeg = np.count_nonzero(neg)
    if numPos > numNeg:
        idxs = np.where(pos.flatten())[0]
        numToZero = numPos-numNeg
    else:
        idxs = np.where(neg.flatten())[0]
        numToZero = numNeg-numPos
    if len(idxs) == 0:
        raise ValueError('something went wrong')
        # return np.zeros_like(mask, dtype=np.uint8)
    toBeZeroed = np.random.choice(idxs, numToZero)
    origShape = mask.shape
    mask = mask.flatten()
    mask[toBeZeroed] = 0
    newMask = mask.reshape(origShape)
    return newMask
