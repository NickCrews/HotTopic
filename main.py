
import time
import hottopic as ht

import cv2

def useGui():
    ht.gui.makeApp()

def predict():
    m = ht.model.FireModel()
    avail = ht.util.availableDays()
    oneDay = ht.rawdata.getDay(*avail[0])
    m.predictDay(oneDay)

def fitPreprocessorWithAugmented():
    trainingDays = trainingDataset()

    aug = ht.augment.Augmentor()
    augmented = trainingDays.copy()
    for _ in range(1):
        augmented.extend(aug.augment(d) for d in trainingDays)

    p = ht.preprocess.PreProcessor()
    p.fit(augmented)
    # p.save('fitWithAugmented.json')

def fitPreprocessorWithoutAugmented():
    trainingDays = trainingDataset()

    p = ht.preprocess.PreProcessor()
    p.fit(trainingDays)
    p.save('fitWithoutAugmented.json')

def trainingDataset():
    allBurnNames = list(ht.util.availableBurnNames())
    #peekaboo, ecklund, redDirt2, redDirt, beaverCreek, riceRidge, junkins, gutzler, coldSprings, pineTree, haydenPass
    testingBurnNames = ['coldSprings', 'riceRidge']
    trainingBurnNames = [b for b in allBurnNames if b not in testingBurnNames]
    trainingDays = []
    for bn in trainingBurnNames:
        trainingDays.extend(ht.rawdata.getAllDays(bn))
    return trainingDays


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
    pre = ht.preprocess.PreProcessor.fromFile('firstFit')
    normed = pre.process(days)
    print(normed)
    samples = ht.sample.makeSamples(normed)
    m.fitOnSamples(samples, epochs=10)
    m.save('NORMEDbc0804')

def testOne():
    m = ht.model.load('bc0804')
    days = [ht.rawdata.getDay('beaverCreek', '0804')]
    pre = ht.preprocess.PreProcessor.fromFile('firstFit')
    normed = pre.process(days)
    print(normed)
    samples = ht.sample.makeSamples(normed)
    renders = ht.viz.render.renderPerformance(m, samples)
    ht.viz.render.show(*renders)

def test():
    m = ht.model.load('secondFit')
    train, test = trainAndTestSets()
    samples = ht.sample.makeSamples(train)
    peekabooSamples = [s for s in samples if s.day.burn.name=='redDirt2']
    # pineTreeSamples = [s for s in samples if s.day.burn.name=='pineTree']
    peekabooRenders = ht.viz.render.renderPerformance(m, peekabooSamples)
    ht.viz.render.show(*peekabooRenders)

def checkForAugmentingAndNorming():
    days = ht.rawdata.getAllDays()
    aug = ht.augment.Augmentor()
    viewer = ht.viz.dayviewer.DayViewer()
    pre = ht.preprocess.PreProcessor.fromFile('fitWithAugmented.json')
    for day in days:
        a = aug.augment(day)
        normed = pre.process(a)
        # viewer.show(a)
        # cv2.imshow("raw", day.layers["dem"])
        # cv2.imshow("augmented", a.layers["dem"])
        # cv2.imshow("normed", normed.layers["dem"])
        cv2.waitKey(0)

if __name__ == '__main__':
    fitPreprocessorWithAugmented()
    # checkForAugmentingAndNorming()
    # ht.conv.trainWithoutAugment(trainingDataset())
    # ht.conv.trainWithAugmentAndWeather(days)
    # ht.conv.test()
    # ht.conv.trainWithAugment()
    # ht.conv.testOnTrainedAugmented()
    # ht.conv.testOnAll()
    # aug = ht.augment.Augmentor()
    # viewer = ht.viz.dayviewer.DayViewer()
    # for day in ht.rawdata.getAllDays():
    #     viewer.show(day)
    #     # viewer = ht.viz.dayviewer.DayViewer(day)
    #     # viewer.show(day)
    #     # ht.viz.render.renderWindRose(day, now=False)
    #     # ht.viz.render.renderWindRose(day, nsector=4, now=True)
    #     new_day = aug.augment(day)
    #     viewer.show(new_day)


# import numpy as np
# def sample():
#     return np.arange()
