
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
    train = []
    test = []
    for (b,d) in ht.util.availableBurnsAndDates():
        if b in testing:
            test.append(ht.rawdata.getDay(b,d))
        else:
            train.append(ht.rawdata.getDay(b,d))
    return train, test

def train():
    m = ht.model.load('secondFit')
    # m.save('firstFit')
    train, test = trainAndTestSets()
    s = ht.sample.makeSamples(train)
    m.fitOnSamples(s, epochs=5)
    m.save('thirdFit')

def trainOnOne():
    m = ht.model.FireModel()
    days = [ht.rawdata.getDay('beaverCreek', '0804')]
    samples = ht.sample.makeSamples(days)
    m.fitOnSamples(samples, epochs=10)
    m.save('bc0804')

def test():
    m = ht.model.load('secondFit')
    train, test = trainAndTestSets()
    samples = ht.sample.makeSamples(train)
    peekabooSamples = [s for s in samples if s.day.burn.name=='redDirt2']
    # pineTreeSamples = [s for s in samples if s.day.burn.name=='pineTree']
    peekabooRenders = ht.viz.renderPerformance(m, peekabooSamples)
    ht.viz.show(*peekabooRenders)

if __name__ == '__main__':
    # test()
    trainOnOne()
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
