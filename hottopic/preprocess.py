import os
import json
import numpy as np
from time import localtime, strftime

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
        assert len(burns) >= 2 and len(days) >= 2

        # Get stats on dems
        print('fitting to dems...', end='\r')
        dems = [b.layers['dem'] for b in burns]
        stdDemRange = np.std([np.nanmax(dem) - np.nanmin(dem) for dem in dems])
        means = [np.nanmean(dem) for dem in dems]
        minDemMean, maxDemMean = np.nanmin(means), np.nanmax(means)
        self.fits = {'dem_std_range':float(stdDemRange),
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
        totPrecips = [ht.util.totalPrecipitation(w) for w in rawWeathers]
        avgHums = [ht.util.averageHumidity(w) for w in rawWeathers]
        maxTemp1s = [ht.util.maximumTemperature1(w) for w in rawWeathers]
        maxTemp2s = [ht.util.maximumTemperature2(w) for w in rawWeathers]
        windSpeeds = [speed for w in rawWeathers for speed in ht.util.windMetrics(w)] #all of the components in a 1d list
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

    def process(self, days, spec=ht.sample.SampleSpec):
        if len(self.fits) == 0:
            raise ValueError('Must fit() before!')
        burns = {}
        results = []
        for day in days:
            # make a new Burn if necessary
            if day.burn in burns:
                newBurn = burns[day.burn]
            else:
                newLayers = self._applyFitToLayers(day.layers, spec)
                oldBurnName = day.burn.name
                newBurnName = oldBurnName + '_processed_' + strftime("%d%b%H_%M", localtime())
                newBurn = ht.rawdata.Burn(newBurnName, newLayers)
                burns[day.burn] = newBurn
            # make the new Day and add it to burn
            newWeather = self._applyFitToWeather(day.weather)
            newDay = ht.rawdata.Day(newBurn, day.date, newWeather, day.layers['starting_perim'], day.layers['ending_perim'])
            newBurn.days[day.date] = newDay
            results.append(newDay)
        return results

    def _applyFitToLayers(self, layers, spec):
        if len(self.fits) == 0:
            raise ValueError('Must fit() before!')
        newLayers = {}
        for name in spec.layers:
            lay = layers[name]
            if name =='starting_perim':
                newLayers[name] = lay
            elif name != 'dem':
                mean = self.fits[name+'_mean']
                std = self.fits[name+'_std']
                newLayer = (lay-mean) / std
                newLayers[name] = newLayer
            else:
                # make the dem be centered around zero, and scaled according to stddev of all dems
                std = self.fits['dem_std_range']
                newDem = (lay-np.nanmean(lay)) / std
                newLayers[name] = newDem
        return newLayers

    def _applyFitToWeather(self, weather):
        if len(self.fits) == 0:
            raise ValueError('Must fit() before!')
        temp, dewpt, temp2, wdir, wspeed, precip, hum = weather
        newTemp1 = (temp - self.fits['max_temp1_mean']) / self.fits['max_temp1_std']
        newTemp2 = (temp - self.fits['max_temp2_mean']) / self.fits['max_temp2_std']
        newTemp1 = (temp - self.fits['max_temp1_mean']) / self.fits['max_temp1_std']
        newWSpeed = (wspeed - self.fits['wind_speeds_mean']) / self.fits['wind_speeds_std']
        newPrecip = (precip - self.fits['total_precip_mean']) / self.fits['total_precip_std']
        newHum = (hum - self.fits['mean_hum_mean']) / self.fits['mean_hum_std']
        return np.array([newTemp1, dewpt, newTemp2, wdir, newWSpeed, newPrecip, newHum])

    # def prepSpatialData(self, day):
    #     normed = self.normalizeLayers(day)
    #     # now order them in the whichLayers order, stack them, and pad them
    #     paddedLayers = stackAndPad(normed, self.AOIRadius)
    #     return paddedLayers
    #
    # def normalizeLayers(self, day):
    #     normed = [day.layers['starting_perim']]
    #     for layerName in self.whichLayers:
    #         # if layerName not in self.fits:
    #         #     raise ValueError('preProcessor was not fitted for the layer {}'.format(layerName))
    #         # if name == 'dem':
    #         #     result[name] = normalizeElevations(data)
    #         # else:
    #         #     # print('normalizing layer', name)
    #         #     result[name] = normalizeNonElevations(data)
    #         layer = day.layers[layerName]
    #         normed.append(layer)
    #     return normed
    #
    # def weatherMetrics(self, day):
    #     rw = day.weather #rawWeather
    #     precip = totalPrecipitation(rw)
    #     temp = maximumTemperature1(rw)
    #     temp2 = maximumTemperature2(rw)
    #     hum = averageHumidity(rw)
    #     winds = windMetrics(rw)
    #     allMetrics = [precip, temp, temp2, hum] + winds
    #     return np.array(allMetrics)

    def __repr__(self):
        return 'PreProcessor({})'.format(self.fits)
