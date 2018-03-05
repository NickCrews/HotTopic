
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

def samples():
    allDays = ht.rawdata.getAllDays()
    s = ht.sample.makeSamples(allDays)
    print('have a total of {} Samples'.format(len(s)))
    for b in ht.sample.getBatches(s, batchSize=5):
        inp = ht.sample.toModelInput(b)
        # print(inp[0].shape, inp[1].shape)
        out = ht.sample.toModelOutput(b)
        # print(out.shape)


# if __name__ == '__main__':
    # fitPreprocessor()
    # usePreprocessor()
    # samples()
    # useGui()
    # predict()


# import numpy as np
# def sample():
#     return np.arange()
