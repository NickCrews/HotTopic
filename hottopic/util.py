
import os

def listdir(path):
    '''List all the files in a path that are not hidden (begin with a .)'''
    def isGood(fname):
        return not fname.startswith('.') and not fname.startswith("_")
    return [f for f in os.listdir(path) if isGood(f)]

def availableBurns():
    return listdir_nohidden('data/')

def availableDates(burnName):
    '''Given a fire, return a list of all dates that we can train on'''
    directory = 'data/{}/'.format(burnName)

    weatherFiles = listdir_nohidden(directory+'weather/')
    weatherDates = [fname[:-len('.csv')] for fname in weatherFiles]

    perimFiles = listdir_nohidden(directory+'perims/')
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
