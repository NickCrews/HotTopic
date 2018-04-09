import os
import json
import numpy as np
from time import localtime, strftime

import numpy as np
import cv2

import hottopic as ht

class PreProcessor(object):

    SCALE_FACTOR = .25
    LAYER_NAMES = ['starting_perim', 'dem', 'ndvi', 'band_2', 'band_3', 'band_4', 'slope']
    WEATHER_NAMES = ['total_precip', 'temp1', 'temp2', 'rel_hum', 'wind_N', 'wind_S', 'wind_E', 'wind_W']
    ALL_INPUTS = LAYER_NAMES + WEATHER_NAMES

    @staticmethod
    def getInput(days, use_weather=True):
        if isinstance(days, ht.rawdata.Day):
            return PreProcessor._getInput(days, use_weather)
        else:
            return [PreProcessor._getInput(day, use_weather) for day in days]

    @staticmethod
    def _getInput(day, use_weather):
        layer_data= PreProcessor.getLayerInputs(day)

        if use_weather:
            metrics = PreProcessor.getWeatherInputs(day)
            # h,w = layer_data.shape[:2]
            # tiled_metrics = np.tile(metrics, [h,w,1])
            # layer_data = np.dstack((layer_data, tiled_metrics))
        else:
            metrics = None

        np.nan_to_num(layer_data, copy=False)
        shrunk = cv2.resize(layer_data, None, fx=PreProcessor.SCALE_FACTOR, fy=PreProcessor.SCALE_FACTOR)
        return shrunk, metrics

    @staticmethod
    def getOutput(days):
        if isinstance(days, ht.rawdata.Day):
            return PreProcessor._getOutput(days)
        else:
            return [PreProcessor._getOutput(day) for day in days]

    @staticmethod
    def _getOutput(day):
        ep = day.layers['ending_perim']
        AOI_bounds = PreProcessor.getAOIBounds(day)
        loy, hiy, lox, hix = AOI_bounds
        cropped = ep[loy:hiy, lox:hix]
        shrunk = cv2.resize(cropped, None, fx=PreProcessor.SCALE_FACTOR, fy=PreProcessor.SCALE_FACTOR)
        shrunk[shrunk <  .5] = 0
        shrunk[shrunk >= .5] = 1.
        # explicit_one_channel = np.expand_dims(shrunk, axis=-1)
        return shrunk

    @staticmethod
    def getWeatherInputs(day):
        weather = day.weather
        # calculate the weather metrics
        totPrecips = ht.util.totalPrecipitation(weather)
        avgHums = ht.util.averageHumidity(weather)
        maxTemp1s = ht.util.maximumTemperature1(weather)
        maxTemp2s = ht.util.maximumTemperature2(weather)
        windSpeeds = ht.util.windMetrics(weather)

        return np.array([totPrecips, avgHums, maxTemp1s, maxTemp2s] + windSpeeds)

    @staticmethod
    def getLayerInputs(day):
        AOI_bounds = PreProcessor.getAOIBounds(day)
        loy, hiy, lox, hix = AOI_bounds

        layers = []
        for name in PreProcessor.LAYER_NAMES:
            layer = day.layers[name]
            cropped = layer[loy:hiy, lox:hix]
            if name =='dem':
                cropped = cropped-np.nanmean(cropped)
            layers.append(cropped)

        return np.dstack(layers)

    @staticmethod
    def getAOIBounds(day):
        ending_perim = day.layers['ending_perim']
        loy,hiy,lox,hix = PreProcessor.getBB(ending_perim)
        # cv2.imshow("ending_perim", ending_perim)
        # expand by radius R (in the final result)
        R = 24
        R = int(R/PreProcessor.SCALE_FACTOR)
        H,W = ending_perim.shape
        loy = max(0, loy-R)
        lox = max(0, lox-R)
        hiy = min(H, hiy+R)
        hix = min(W, hix+R)
        return loy,hiy,lox,hix

    @staticmethod
    def reconstruct_output(output, day):
        '''Given an output from the model and the day it was predicted from,
        scale the output back up so that it actually fits over the original day

        re-enlarge the output and re-insert it in the actual day it was exptracted from'''
        loy,hiy,lox,hix = PreProcessor.getAOIBounds(day)
        bigger = cv2.resize(output, (hix-lox, hiy-loy))
        canvas = np.zeros_like(day.layers['starting_perim'])
        canvas[loy:hiy, lox:hix] = bigger
        return canvas

    @staticmethod
    def getBB(ending_perim):
        h,w = ending_perim.shape
        noNans = np.nan_to_num(ending_perim, copy=True)
        col_contains_nonzero = noNans.any(axis=0)
        row_contains_nonzero = noNans.any(axis=1)
        loy = np.argmax(row_contains_nonzero)
        hiy = h-np.argmax(row_contains_nonzero[::-1])
        lox = np.argmax(col_contains_nonzero)
        hix = w-np.argmax(col_contains_nonzero[::-1])
        return loy,hiy,lox,hix

    def __repr__(self):
        return 'PreProcessor({})'
