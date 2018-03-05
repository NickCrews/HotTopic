
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
        # yes will contain all 'did burn' points, no contains 'did not burn' points
        yes = []
        no =  []
        for p in self.toList(self.masks):
            burnName, date, loc = p
            burn = self.data.burns[burnName]
            day = burn.days[date]
            out = day.endingPerim[loc]
            if out:
                yes.append(p)
            else:
                no.append(p)
        # shorten whichever is longer
        if len(yes) > len(no):
            random.shuffle(yes)
            yes = yes[:len(no)]
        else:
            random.shuffle(no)
            no = no[:len(yes)]

        # recombine
        return self.toDict(yes+no)
