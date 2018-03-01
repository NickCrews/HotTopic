
import numpy as np

import hottopic as ht

class PreProcessor(object):

    def __init__(self, numWeatherInputs, whichLayers, AOIRadius):
        self.numWeatherInputs = numWeatherInputs
        self.whichLayers = whichLayers
        self.AOIRadius = AOIRadius

        self.fits = {}

    def fit(self, dataset):
        pass

    def processDay(self, day):
        sd = self.prepSpatialData(day)
        outs = day.endingPerim

        wm = self.weatherMetrics(day)
        H,W = outs.shape
        wmtiled = np.tile(wm, (H,W,1))
        return [wmtiled, sd], outs

    def prepSpatialData(self, day):
        normed = self.normalizeLayers(day)
        # now order them in the whichLayers order, stack them, and pad them
        paddedLayers = stackAndPad(normed, self.AOIRadius)
        return paddedLayers

    def normalizeLayers(self, day):
        normed = [day.startingPerim]
        for layerName in self.whichLayers:
            # if layerName not in self.fits:
            #     raise ValueError('preProcessor was not fitted for the layer {}'.format(layerName))
            # if name == 'dem':
            #     result[name] = normalizeElevations(data)
            # else:
            #     # print('normalizing layer', name)
            #     result[name] = normalizeNonElevations(data)
            layer = day.burn.layers[layerName]
            normed.append(layer)
        return normed

    def weatherMetrics(self, day):
        rw = day.weather #rawWeather
        precip = totalPrecipitation(rw)
        temp = maximumTemperature1(rw)
        temp2 = maximumTemperature2(rw)
        hum = averageHumidity(rw)
        winds = windMetrics(rw)
        allMetrics = [precip, temp, temp2, hum] + winds
        return np.array(allMetrics)


def stackAndPad(layers, borderRadius):
    stacked = np.dstack(layers)
    r = borderRadius
    # pad with zeros around border of image
    padded = np.lib.pad(stacked, ((r,r),(r,r),(0,0)), 'constant')
    return padded

# =================================================================
# weather metric utility functions

def totalPrecipitation(weatherMatrix):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherMatrix
    return sum(precip)

def averageHumidity(weatherMatrix):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherMatrix
    return sum(hum)/len(hum)

def maximumTemperature1(weatherMatrix):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherMatrix
    return max(temp)

def maximumTemperature2(weatherMatrix):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherMatrix
    return max(temp2)

def windMetrics(weatherMatrix):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = weatherMatrix
    wDirRad = [(np.pi/180) * wDirDeg for wDirDeg in wdir]
    n, s, e, w = 0, 0, 0, 0
    for hr in range(len(wdir)):
        # print(wdir[i], wDirRad[i], wspeed[i])
        if wdir[hr] > 90 and wdir[hr] < 270: #from south
            # print('south!', -np.cos(wDirRad[i]) * wspeed[i])
            s += abs(np.cos(wDirRad[hr]) * wspeed[hr])
        if wdir[hr] < 90 or wdir[hr] > 270: #from north
            # print('north!', np.cos(wDirRad[i]) * wspeed[i])
            n += abs(np.cos(wDirRad[hr]) * wspeed[hr])
        if wdir[hr] < 360 and wdir[hr] > 180: #from west
            # print('west!', -np.sin(wDirRad[i]) * wspeed[i])
            w += abs(np.sin(wDirRad[hr]) * wspeed[hr])
        if wdir[hr] > 0 and wdir[hr] < 180: #from east
            # print('east!',np.sin(wDirRad[i]) * wspeed[i])
            e += abs(np.sin(wDirRad[hr]) * wspeed[hr])
    components = [n, s, e, w]
    # print(weather)
    return components
