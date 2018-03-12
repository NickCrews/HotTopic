
import os
import numpy as np
import cv2

def listdir(path):
    '''List all the files in a path that are not hidden (begin with a .)'''
    def isGood(fname):
        return not fname.startswith('.') and not fname.startswith("_")
    for f in os.listdir(path):
        if isGood(f):
            yield f

def normalize(arr, axis=None):
    '''Rescale an array so that it varies from 0-1.

    if axis=0, then each column is normalized independently
    if axis=1, then each row is normalized independently'''
    arr = arr.astype(np.float32)
    res = arr - np.nanmin(arr, axis=axis)
    # where dividing by zero, just use zero
    res = np.divide(res, np.nanmax(res, axis=axis), out=np.zeros_like(res), where=res!=0)
    return res

def openImg(fname):
    if not os.path.exists(fname):
        raise FileNotFoundError("The file {} does not exist.".format(fname))
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not open the file {} as an image".format(fname))
    img = img.astype(np.float32)
    # go through all the channels
    channels = cv2.split(img)
    for c in channels:
        # find any "invalid" pixels and set them to nan, so we can find them easily later
        c[invalidPixelIndices(c)] = np.nan
    return cv2.merge(channels)

def openPerim(fname):
    if not os.path.exists(fname):
        raise FileNotFoundError("The file {} does not exist.".format(fname))
    img = cv2.imread(fname, 0)
    if img is None:
        raise ValueError("Could not open the file {} as an image".format(fname))
    img = img.astype(np.uint8)
    if len(img.shape)>2:
        img = img[:,:,0]
    return img

def validPixelIndices(layer):
    validPixelMask = 1-invalidPixelMask(layer)
    return np.where(validPixelMask)

def invalidPixelIndices(layer):
    return np.where(invalidPixelMask(layer))

def invalidPixelMask(layer):
    # If there are any massively valued pixels, just return those
    HUGE = 1e10
    huge = np.absolute(layer) > HUGE
    if np.any(huge):
        return huge

    # floodfill in from every corner, all the NODATA pixels are the same value so they'll get found
    h,w = layer.shape[:2]
    noDataMask = np.zeros((h+2,w+2), dtype = np.uint8)
    fill = 1
    seeds = [(0,0), (0,h-1), (w-1,0), (w-1,h-1)]
    for seed in seeds:
        cv2.floodFill(layer.copy(), noDataMask, seed, fill)
        # plt.figure('layer')
        # plt.imshow(layer)
        # plt.figure('noDataMask')
        # plt.imshow(noDataMask)
        # plt.show()

    # extract ouf the center of the mask, which corresponds to orig image
    noDataMask = noDataMask[1:h+1, 1:w+1]
    return noDataMask

def availableBurnNames():
    yield from listdir('data/')

def availableDates(burnName):
    '''Given a fire, return a list of all dates that we can train on'''
    directory = 'data' + os.sep + burnName + os.sep

    weatherFiles = listdir(directory+'weather' + os.sep)
    weatherDates = [fname[:-len('.csv')] for fname in weatherFiles]

    perimFiles = listdir(directory+'perims' + os.sep)
    perimDates = [fname[:-len('.tif')] for fname in perimFiles]
    perimDates.sort()
    # we can only use days which have perimeter and weather data on the following day
    for d in perimDates:
        a, b = possibleNextDates(d)
        if (a in perimDates and a in weatherDates) or (b in perimDates and b in weatherDates):
            yield d

def availableBurnsAndDates():
    for burnName in availableBurnNames():
        for date in availableDates(burnName):
            yield (burnName, date)

def possibleNextDates(dateString):
    month, day = dateString[:2], dateString[2:]

    nextDay = str(int(day)+1).zfill(2)
    guess1 = month+nextDay

    nextMonth = str(int(month)+1).zfill(2)
    guess2 = nextMonth+'01'
    return guess1, guess2

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
