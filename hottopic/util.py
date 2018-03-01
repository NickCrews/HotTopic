
import os
import numpy as np
import cv2

def listdir(path):
    '''List all the files in a path that are not hidden (begin with a .)'''
    def isGood(fname):
        return not fname.startswith('.') and not fname.startswith("_")
    return [f for f in os.listdir(path) if isGood(f)]

def isValidImg(fname):
    img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    return img is not None

def normalize(arr, axis=None):
    '''Rescale an array so that it varies from 0-1.

    if axis=0, then each column is normalized independently
    if axis=1, then each row is normalized independently'''

    arr = arr.astype(np.float32)
    # print('subtracting min')
    res = arr - np.nanmin(arr, axis=axis)
    # print('dividing where', res)
    # where dividing by zero, just use zero
    res = np.divide(res, np.nanmax(res, axis=axis), out=np.zeros_like(res), where=res!=0)
    # print('done')
    return res

def openImg(fname):
    if not os.path.exists(fname):
        raise ValueError("The file {} does not exist.".format(fname))
    if "/perims/" in fname:
        img = cv2.imread(fname, 0)
    else:
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    try:
        img = img.astype(np.float32)
    except AttributeError:
        raise ValueError("Could not open the file {} as an image".format(fname))

    # go through all the channels
    channels = cv2.split(img)
    for c in channels:
        # find any "invalid" pixels and set them to nan, so we can find them easily later
        c[invalidPixelIndices(c)] = np.nan
    return cv2.merge(channels)

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

def availableBurns():
    return listdir('data/')

def availableDates(burnName):
    '''Given a fire, return a list of all dates that we can train on'''
    directory = 'data/{}/'.format(burnName)

    weatherFiles = listdir(directory+'weather/')
    weatherDates = [fname[:-len('.csv')] for fname in weatherFiles]

    perimFiles = listdir(directory+'perims/')
    perimDates = [fname[:-len('.tif')] for fname in perimFiles if isValidImg(directory+'perims/'+fname)]

    # we can only use days which have perimeter data on the following day
    daysWithFollowingPerims = []
    for d in perimDates:
        nextDay1, nextDay2 = possibleNextDates(d)
        if nextDay1 in perimDates or nextDay2 in perimDates:
            daysWithFollowingPerims.append(d)

    # now we have to verify that we have weather for these days as well
    daysWithWeatherAndPerims = [d for d in daysWithFollowingPerims if d in weatherDates]
    daysWithWeatherAndPerims.sort()
    return daysWithWeatherAndPerims

def availableDays():
    avail = []
    for burnName in availableBurns():
        for date in availableDates(burnName):
            avail.append((burnName, date))
    return avail

def possibleNextDates(dateString):
    month, day = dateString[:2], dateString[2:]

    nextDay = str(int(day)+1).zfill(2)
    guess1 = month+nextDay

    nextMonth = str(int(month)+1).zfill(2)
    guess2 = nextMonth+'01'
    return guess1, guess2
