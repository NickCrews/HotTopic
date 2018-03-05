
import numpy as np
import cv2

VULNERABLE_RADIUS = 750 #in meters

class Sample(object):
    def __init__(self, sampleSpec, day, loc):
        self.sampleSpec = sampleSpec
        self.day = day
        self.loc = loc

class SampleSpec(object):
    AOIRadius = 30
    AOISize = (2*AOIRadius+1, 2*AOIRadius+1)
    layerNames = ['start_perim', 'dem', 'ndvi']
    numLayers = len(layerNames)
    nonLayers = ['rel_hum', 'temp', 'dew_pt']
    numNonLayers = len(nonLayers)

def vulnerablePixels(day, radius=VULNERABLE_RADIUS):
    '''Return a boolean mask of the pixels that are close to the current fire perimeter. radius is in meters.'''
    startingPerim = day.startingPerim.astype(np.uint8)
    kernel = np.ones((3,3))
    its = int(round((2*(radius/rawdata.PIXEL_SIZE)**2)**.5))
    dilated = cv2.dilate(startingPerim, kernel, iterations=its)
    border = dilated - startingPerim
    return border.astype(np.uint8)

def evenOutPositiveAndNegativeOld(day, mask):
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
        continue
    toBeZeroed = np.random.choice(idxs, numToZero)
    origShape = mask.shape
    mask = mask.flatten()
    mask[toBeZeroed] = 0
    newMask = mask.reshape(origShape)
    return newMask
