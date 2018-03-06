
import hottopic as ht

def useGui():
    ht.gui.makeApp()

def predict():
    m = ht.model.FireModel()
    avail = ht.util.availableDays()
    oneDay = ht.rawdata.getDay(*avail[0])
    m.predictDay(oneDay)

def fitPreprocessor():
    p = ht.preprocess.PreProcessor()
    allDays = ht.rawdata.getAllDays()
    p.fit(allDays)
    p.save('firstFit.json')

def usePreprocessor():
    p = ht.preprocess.PreProcessor.fromFile('firstFit')
    print(p)
    allDays = ht.rawdata.getAllDays()
    res = p.process(allDays)
    # print(res)
    for newDay in res:
        print(newDay.weather)
        print(newDay.burn.layers['dem'])

def example():
    allDays = ht.rawdata.getAllDays()
    s = ht.sample.makeSamples(allDays)
    print('have a total of {} Samples'.format(len(s)))
    m = ht.model.FireModel()
    # for b in ht.sample.getBatches(s, batchSize=32):
    #     inp = ht.sample.toModelInput(b)
    #     print(inp[0].shape, inp[1].shape)
    #     out = ht.sample.toModelOutput(b)
    #     m.fit(inp, out)
        # print(out.shape)
    m.fitOnSamples(s)

def trainAndTestSets():
    testing = ['peekaboo', 'pineTree']
    train = [ht.rawdata.getDay(b,d) for (b,d) in ht.util.availableBurnsAndDates() if b not in testing]
    test = [ht.rawdata.getDay(b,d) for (b,d) in ht.util.availableBurnsAndDates() if b in testing]
    return train, test

def train():
    m = ht.model.FireModel()
    m.save('firstFit')
    train, test = trainAndTestSets()
    s = ht.sample.makeSamples(train)
    m.fitOnSamples(s)
    m.save('firstFit')

if __name__ == '__main__':
    train()
    # fitPreprocessor()
    # usePreprocessor()
    # samples()
    # useGui()
    # predict()
    # for pair in ht.util.availableBurnsAndDates():
    #     print(pair)


# import numpy as np
# def sample():
#     return np.arange()
