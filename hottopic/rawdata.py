
import os
import numpy as np
import cv2

import hottopic as ht

_burns = {}
_days = {}

def getBurn(burnName):
    if burnName in _burns:
        return _burns[burnName]
    else:
        burn = Burn(burnName)
        _burns[burnName] = burn
        return burn

def getDay(burnName, date):
    if (burnName, date) in _days:
        return _days[(burnName, date)]
    # we havent created this day yet.
    # do we need to create the burn for it?
    if burnName in _burns:
        burn = _burns[burnName]
    else:
        burn = Burn(burnName)
        _burns[burnName] = burn

    day = Day(burn, date)
    _days[(burnName, date)] = day
    return day

class Burn(object):

    def __init__(self, name):
        self.name = name
        self.layers = self.loadLayers()
        # what is the height and width of a layer of data
        self.layerSize = list(self.layers.values())[0].shape[:2]

    def loadLayers(self):
        folder = 'data' + os.sep + self.name + os.sep
        dem =    ht.util.openImg(folder+'dem.tif')
        # slope = util.openImg(folder+'slope.tif')
        band_2 = ht.util.openImg(folder+'band_2.tif')
        band_3 = ht.util.openImg(folder+'band_3.tif')
        band_4 = ht.util.openImg(folder+'band_4.tif')
        band_5 = ht.util.openImg(folder+'band_5.tif')
        ndvi =   ht.util.openImg(folder+'ndvi.tif')
        aspect = ht.util.openImg(folder+'aspect.tif')
        # r,g,b,nir = cv2.split(landsat)

        layers = {'dem':dem,
                # 'slope':slope,
                'ndvi':ndvi,
                'aspect':aspect,
                'band_4':band_4,
                'band_3':band_3,
                'band_2':band_2,
                'band_5':band_5}

        return layers

    def getDates(self):
        return ht.util.availableDates(self.name)

    def __repr__(self):
        return "Burn({}, {})".format(self.name, [d.date for d in self.days.values()])

class Day(object):

    def __init__(self, burn, date):
        self.burn = burn
        self.date = date
        self.weather = self.loadWeather()
        self.startingPerim = self.loadStartingPerim()
        self.endingPerim = self.loadEndingPerim()

    def loadWeather(self):
        fname = 'data' + os.sep + self.burn.name + os.sep + 'weather' + os.sep + self.date+'.csv'
        # the first row is the headers, and only cols 4-11 are actual data
        data = np.loadtxt(fname, skiprows=1, usecols=range(5,12), delimiter=',').T
        # now data is 2D array
        return data

    def loadStartingPerim(self):
        fname =  'data' + os.sep + self.burn.name + os.sep + 'perims' + os.sep + self.date+'.tif'
        perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if perim is None:
            raise RuntimeError('Could not find a perimeter for the fire {} for the day {}'.format(self.burn.name, self.date))
        perim[perim!=0] = 255
        return perim

    def loadEndingPerim(self):
        guess1, guess2 = ht.util.possibleNextDates(self.date)
        fname = 'data' + os.sep + self.burn.name + os.sep + 'perims' + os.sep + guess1+'.tif'
        perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        if perim is None:
            # overflowed the month, that file didnt exist
            fname = 'data' + os.sep + self.burn.name + os.sep + 'perims' + os.sep + guess2+'.tif'
            perim = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            if perim is None:
                raise RuntimeError('Could not open a perimeter for the fire {} for the day {} or {}'.format(self.burn.name, guess1, guess2))
        return perim

    def __repr__(self):
        return "Day({},{})".format(self.burn.name, self.date)
