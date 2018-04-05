import numpy as np
import cv2

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Concatenate, Input
from keras.optimizers import SGD, RMSprop
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

import hottopic as ht

SCALE_FACTOR = .25

def make_model(sample_spec):
    kernel_size=(5,5)
    # one smaple at a time, dont know the H and W, one channel
    inp_shape = (None,None,sample_spec.numLayers)

    m = Sequential()
    m.add(Conv2D(4, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=inp_shape))
    m.add(Conv2D(8, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    m.add(Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    # m.add(Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    m.summary()
    return m

def make_model_with_weather(sample_spec):
    kernel_size=(5,5)
    # one smaple at a time, dont know the H and W, one channel
    nchannels = sample_spec.numLayers + sample_spec.numNonLayers
    inp_shape = (None,None,nchannels)

    m = Sequential()
    m.add(Conv2D(16, (1,1), strides=(1,1), padding='same', data_format='channels_last', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=inp_shape))

    m.add(Conv2D(16, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.add(Conv2D(16, (1,1), strides=(1,1), padding='same', data_format='channels_last', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.add(Conv2D(32, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.add(Conv2D(32, (1,1), strides=(1,1), padding='same', data_format='channels_last', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.add(Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    # m.add(Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    m.summary()
    return m

def make_generator(days, normalizer, augmentor=None, use_weather=False):
    pp = ht.preprocess.PreProcessor
    def generator():
        while True:
            for day in days:
                if augmentor:
                    day = augmentor.augment(day)
                inp = np.expand_dims(pp.getInput(day, use_weather), axis=0)
                out = np.expand_dims(pp.getOutput(day), axis=0)
                # for i in range(inp.shape[-1]):
                #     print(i, inp[:,:,i].mean())
                # assert np.isnan(inp).any() == False
                # assert np.isnan(out).any() == False
                normed = normalizer.normalize(inp)
                yield normed, out
    return generator

def fitPreprocessorWithAugmentedAndWeather(days):
    days = days[:3]
    aug = ht.augment.Augmentor()
    augmented = []
    for _ in range(1):
        augmented.extend(aug.augment(d) for d in days)

    inputs = [ht.preprocess.PreProcessor.getInput(aug) for aug in augmented]

    normer = ht.normalizer.Normalizer()
    normer.fit(inputs)
    normer.save('normerApril4.json')

def trainWithoutAugment(days):
    pre = ht.preprocess.PreProcessor.fromFile('fitWithoutAugmented.json')
    gen = make_generator(pre, days, augmentor=None)
    train(gen, 'models/convWithoutAugmented.h5')

def trainWithAugment(days):
    pre = ht.preprocess.PreProcessor.fromFile('fitWithAugmented.json')
    aug = ht.augment.Augmentor()
    gen = make_generator(pre, days, aug)
    train(gen, 'models/convWithAugmented.h5')

def trainWithAugmentAndWeather(days):
    pre = ht.preprocess.PreProcessor.fromFile('fitWithAugmented.json')
    aug = ht.augment.Augmentor()
    gen = make_generator(pre, days, aug, use_weather=True)
    trainWithWeather(gen, 'models/convWithAugmentedAndWeather.h5')

def testOn(days):
    pre = ht.preprocess.PreProcessor.fromFile('fitWithAugmented.json')
    normed = pre.process(days)
    # m = keras.models.load_model('models/convWithAugmentedBig.h5')
    m = keras.models.load_model('models/secondConv.h5')
    for day in normed:
        inp, expected = np.expand_dims(getInputs(day),axis=0), np.expand_dims(getOutput(day), axis=0)
        assert np.isnan(inp).any() == False
        assert np.isnan(expected).any() == False
        out = m.predict(inp)
        assert np.isnan(out).any() == False
        # inp = np.squeeze(inp)
        out = np.squeeze(out)
        # cv2 uses (w,h) and numpy uses (h,w) WTF
        h,w = day.layers['starting_perim'].shape
        out = cv2.resize(out, (w,h))
        canvas = ht.viz.render.renderCanvas(day)
        viz = ht.viz.render.overlayPredictions(canvas, out)
        cv2.imshow(day.burn.name+day.date, viz)
        if cv2.waitKey(0) == ord('q'):
            break
        cv2.destroyWindow(day.burn.name+day.date)

def testOnTrainedAugmentedAndWeather(days):
    pre = ht.preprocess.PreProcessor.fromFile('fitWithAugmentedAfterFixZooming.json')
    augmentor = ht.augment.Augmentor()
    auged = [augmentor.augment(day) for day in days]
    normed = pre.process(auged)
    # m = keras.models.load_model('models/convWithAugmentedBig.h5')
    m = keras.models.load_model('models/convWithAugmentedAndWeather.h5')
    for day in normed:
        inp, expected = np.expand_dims(getInputsWithWeather(day, pre),axis=0), np.expand_dims(getOutput(day), axis=0)
        assert np.isnan(inp).any() == False
        assert np.isnan(expected).any() == False
        out = m.predict(inp)
        assert np.isnan(out).any() == False
        # inp = np.squeeze(inp)
        out = np.squeeze(out)
        # cv2 uses (w,h) and numpy uses (h,w) WTF
        h,w = day.layers['starting_perim'].shape
        out = cv2.resize(out, (w,h))
        canvas = ht.viz.render.renderCanvas(day)
        viz = ht.viz.render.overlayPredictions(canvas, out)
        cv2.imshow(day.burn.name+day.date, viz)
        if cv2.waitKey(0) == ord('q'):
            break
        cv2.destroyWindow(day.burn.name+day.date)

def testOnTrainedAugmented():
    train = trainingDataset()
    testOn(train)

def testOnAll():
    allDays = ht.rawdata.getAllDays()
    testOn(allDays)

def train(generator, model_name):
    m = make_model(ht.sample.SampleSpec)
    m.fit_generator(generator(), steps_per_epoch=16, epochs=256)
    m.save(model_name)

def trainWithWeather(generator, model_name):
    m = make_model_with_weather(ht.sample.SampleSpec)
    m.fit_generator(generator(), steps_per_epoch=16, epochs=64)
    m.save(model_name)

def test():
    m = keras.models.load_model('models/secondConv.h5')
    test = ht.rawdata.chooseDays()
    pre = ht.preprocess.PreProcessor.fromFile('secondConvFit')
    normed = pre.process(test)
    for day in normed:
        inp, expected = np.expand_dims(getInputs(day),axis=0), np.expand_dims(getOutput(day), axis=0)
        assert np.isnan(inp).any() == False
        assert np.isnan(expected).any() == False
        out = m.predict(inp)
        assert np.isnan(out).any() == False
        # inp = np.squeeze(inp)
        out = np.squeeze(out)
        # cv2 uses (w,h) and numpy uses (h,w) WTF
        h,w = day.layers['starting_perim'].shape
        out = cv2.resize(out, (w,h))
        canvas = ht.viz.render.renderCanvas(day)
        viz = ht.viz.render.overlayPredictions(canvas, out)
        cv2.imshow('viz', viz)
        # expected = np.squeeze(expected)
        # print(inp.dtype, expected.dtype, out.dtype)
        # print(inp.shape, expected.shape, out.shape)
        # start = inp[:,:,0]
        # print(start.dtype, start.min(), start.max())
        # out = np.round(np.clip(out,0,1)).astype(np.uint8)
        #
        # canvas = np.zeros_like(start, dtype=np.uint8)
        # canvas[np.where(out)] = 90
        # canvas[np.where(expected)] = 180
        # canvas[np.where(start)] = 255
        # cv2.imshow('canvas', canvas)

        # cv2.imshow('start', start)
        # cv2.imshow('exp', expected*255)
        # cv2.imshow('out', out*255)
        # cv2.imshow('diff', np.bitwise_xor(out,expected)*255)
        cv2.waitKey(0)
