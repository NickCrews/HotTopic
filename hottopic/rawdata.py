
import os
import numpy as np
import cv2

import hottopic as ht

PIXEL_SIZE = 30 #in meters

_burns = {}
_days = {}

def getBurn(burnName):
    if burnName in _burns:
        return _burns[burnName]
    else:
        burn = Burn.fromFile(burnName)
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
        burn = Burn.fromFile(burnName)
        _burns[burnName] = burn

    day = Day.fromFile(burn, date)
    _days[(burnName, date)] = day
    return day

def getAllDays():
    for burnName, date in ht.util.availableBurnsAndDates():
        yield getDay(burnName, date)

def getAllBurns():
    for burnName in ht.util.availableBurnNames():
        yield getBurn(burnName)

class Burn(object):

    def __init__(self, name, layers):
        self.name = name
        self.layers = layers
        # what is the height and width of a layer of data
        self.days = {}

    @staticmethod
    def fromFile(burnName):
        print('loading Burn {}...'.format(burnName))
        l = Burn.loadLayers(burnName)
        b = Burn(burnName,l)
        days = {date:Day.fromFile(b, date) for date in ht.util.availableDates(burnName)}
        b.days = days
        print('loading Burn {}...done'.format(burnName))
        return b

    @staticmethod
    def loadLayers(burnName):
        folder = 'data' + os.sep + burnName + os.sep
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

    def __repr__(self):
        return "Burn({}, [{}])".format( self.name, sorted(self.days.keys()) )

class Day(object):

    def __init__(self, burn, date, weather, starting_perim, ending_perim):
        self.burn = burn
        self.date = date
        self.weather = weather
        self.layers = {'starting_perim': starting_perim, 'ending_perim':ending_perim}
        self.layers.update(self.burn.layers)

    @staticmethod
    def fromFile(burn, date):
        print('loading Day {}, {}...'.format(burn.name, date), end='\r')
        w = Day.loadWeather(burn.name, date)
        sp = Day.loadStartingPerim(burn.name, date)
        ep = Day.loadEndingPerim(burn.name, date)
        print('loading Day {}, {}...done'.format(burn.name, date))
        return Day(burn, date, w, sp, ep)

    @staticmethod
    def loadWeather(burnName, date):
        fname = 'data' + os.sep + burnName + os.sep + 'weather' + os.sep + date+'.csv'
        # the first row is the headers, and only cols 4-11 are actual data
        data = np.loadtxt(fname, skiprows=1, usecols=range(5,12), delimiter=',').T
        # now data is 2D array
        return data

    @staticmethod
    def loadStartingPerim(burnName, date):
        fname =  'data' + os.sep + burnName + os.sep + 'perims' + os.sep + date+'.tif'
        perim = ht.util.openPerim(fname)
        perim[perim!=0] = 255
        return perim

    @staticmethod
    def loadEndingPerim(burnName, date):
        guess1, guess2 = ht.util.possibleNextDates(date)
        directory = 'data' + os.sep + burnName + os.sep + 'perims' + os.sep
        try:
            perim = ht.util.openPerim(directory + guess1 + '.tif')
        except FileNotFoundError:
            # overflowed the month, that file didnt exist
            perim = ht.util.openPerim(directory + guess2 + '.tif')
        return perim

    def __repr__(self):
        return "Day({},{})".format(self.burn.name, self.date)
