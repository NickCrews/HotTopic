
import random

import numpy as np
import cv2

import hottopic as ht

VULNERABLE_RADIUS = 750 #in meters
BATCH_SIZE = 32 #number of samples per batch of training

class Sample(object):

    stackedAndPadded = {}
    weatherMetrics = {}

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
        if self.day in Sample.weatherMetrics:
            return Sample.weatherMetrics[self.day]
        else:
            rw = self.day.weather #rawWeather
            precip = ht.util.totalPrecipitation(rw)
            temp = ht.util.maximumTemperature1(rw)
            temp2 = ht.util.maximumTemperature2(rw)
            hum = ht.util.averageHumidity(rw)
            winds = ht.util.windMetrics(rw)
            allMetrics = [precip, temp, temp2, hum] + winds
            result = np.array(allMetrics)
            Sample.weatherMetrics[self.day] = result
            return result

    def getOutput(self):
        return self.day.layers['ending_perim'][self.loc]

    def stackAndPad(self):
        layers = [self.day.layers[name] for name in self.spec.layers]
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
    layers = ['starting_perim', 'dem', 'ndvi', 'band_2', 'band_3', 'band_4', 'aspect']
    numLayers = len(layers)
    nonLayers = ['total_precip', 'temp1', 'temp2', 'rel_hum', 'wind_N', 'wind_S', 'wind_E', 'wind_W']
    numNonLayers = len(nonLayers)

def makeSamples(days, spec=SampleSpec, doFilter=True):
    samples = []
    for day in days:
        if doFilter:
            vulnerable = vulnerablePixels(day)
            mask = evenOutPositiveAndNegative(day, vulnerable)
        else:
            # use all the points
            mask = np.ones_like(day.layers['starting_perim'], dtype=np.uint8)
        ys, xs =  np.where(mask)
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
    nonSpatials = np.array([sample.getNonSpatialData() for sample in batch])
    spatials = np.array([sample.getSpatialData() for sample in batch])
    return [nonSpatials, spatials]

def toModelOutput(batch):
    return np.array([sample.getOutput() for sample in batch])

def generateTrainingData(samples, batchSize=BATCH_SIZE, shuffle=True):
    while True:
        for batch in getBatches(samples, batchSize, shuffle):
            inp = toModelInput(batch)
            out = toModelOutput(batch)
            yield (inp, out)

def vulnerablePixels(day, radius=VULNERABLE_RADIUS):
    '''Return a boolean mask of the pixels that are close to the current fire perimeter. radius is in meters.'''
    startingPerim = day.layers['starting_perim'].astype(np.uint8)
    kernel = np.ones((3,3))
    its = int(round((2*(radius/ht.rawdata.PIXEL_SIZE)**2)**.5))
    dilated = cv2.dilate(startingPerim, kernel, iterations=its)
    border = dilated - startingPerim
    return border.astype(np.uint8)

def evenOutPositiveAndNegative(day, mask):
    '''Make it so the chosen pixels for a day is a more even mixture of yes and no outputs.
    day is the Day to look at. mask is the boolean mask of locations we are looking at now'''
    didBurn = day.layers['ending_perim'].astype(np.uint8)
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
