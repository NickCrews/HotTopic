import os
import json
import numpy as np

import hottopic as ht

class PreProcessor(object):

    def __init__(self, fits=None):
        if fits is None:
            fits = {}
        self.fits = fits

    def fit(self, days):
        '''Generate the parameters to preprocess this group of days'''
        days = list(days)
        print('Fitting to days {}'.format(days))
        burns = []
        for d in days:
            if d.burn not in burns:
                burns.append(d.burn)
        print('That is {} days within {} Burns'.format(len(days), len(burns)))

        # Get stats on dems
        print('fitting to dems...', end='\r')
        dems = [b.layers['dem'] for b in burns]
        maxDemRange = max(np.nanmax(dem) - np.nanmin(dem) for dem in dems)
        means = [np.nanmean(dem) for dem in dems]
        minDemMean, maxDemMean = np.nanmin(means), np.nanmax(means)
        self.fits = {'dem_max_range':float(maxDemRange),
                    'dem_min_mean':float(minDemMean),
                    'dem_max_mean':float(maxDemMean)}
        print('fitting to dems...done')

        # get stats on the rest of the spatial data
        print('fitting to spatial data...', end='\r')
        layerNamesOfBurns = [sorted(b.layers.keys()) for b in burns]
        assert all(layerNamesOfBurns[0] == ln for ln in layerNamesOfBurns) # ensure all burns have the same layers
        layerNames = set(layerNamesOfBurns[0])-{'dem'} #get one set of layer names, minus the dems
        for name in layerNames:
            data = np.concatenate([b.layers[name].flatten() for b in burns])
            mean = np.nanmean(data)
            devs = np.nanstd(data)
            self.fits[name+'_mean'] = float(mean)
            self.fits[name+'_std'] = float(devs)
        print('fitting to spatial data...done')


        # get stats on the weather data
        print('fitting to weather data...', end='\r')
        rawWeathers = [day.weather for day in days]
        totPrecips = [totalPrecipitation(w) for w in rawWeathers]
        avgHums = [averageHumidity(w) for w in rawWeathers]
        maxTemp1s = [maximumTemperature1(w) for w in rawWeathers]
        maxTemp2s = [maximumTemperature2(w) for w in rawWeathers]
        windSpeeds = [speed for w in rawWeathers for speed in windMetrics(w)] #all of the components in a 1d list
        keys = ['total_precip', 'mean_hum', 'max_temp1', 'max_temp2', 'wind_speeds']
        metrics = [totPrecips, avgHums, maxTemp1s, maxTemp2s, windSpeeds]
        for key, metric in zip(keys, metrics):
            self.fits[key+'_mean'] = np.mean(metric)
            self.fits[key+'_std']  = np.std(metric)
        print('fitting to weather data...done')


    def save(self, fname):
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

    def __repr__(self):
        return 'PreProcessor({})'.format(self.fits)


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
