

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
