
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



if __name__ == '__main__':
    # example()
    # fitPreprocessor()
    # usePreprocessor()
    # samples()
    # useGui()
    # predict()
    for pair in ht.util.availableBurnsAndDates():
        print(pair)


# import numpy as np
# def sample():
#     return np.arange()
