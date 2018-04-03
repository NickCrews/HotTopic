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

def getWeatherInputs(day):
    rw = day.weather #rawWeather
    precip = ht.util.totalPrecipitation(rw)
    temp = ht.util.maximumTemperature1(rw)
    temp2 = ht.util.maximumTemperature2(rw)
    hum = ht.util.averageHumidity(rw)
    winds = ht.util.windMetrics(rw)
    allMetrics = [precip, temp, temp2, hum] + winds
    return np.asarray(allMetrics)

def getInputsWithWeather(day, spec=ht.sample.SampleSpec):
    layers = [day.layers[name] for name in spec.layers]
    stacked = np.dstack(layers)

    metrics = getWeatherInputs(day)
    h,w = layers[0].shape
    tiled_metrics = np.tile(metrics, [h,w,1])

    all_stacked = np.dstack((stacked, tiled_metrics))

    np.nan_to_num(all_stacked, copy=False)
    # for i in range(spec.numLayers):
    #     layer = stacked[:,:,i]
    #     print(spec.layers[i],layer.dtype, layer.shape, layer.min(), layer.max(), layer.mean(), layer.std())
    #     if layer.dtype == np.uint8:
    #         cv2.imshow('perim', layer*255)
    #     else:
    #         cv2.imshow(spec.layers[i],layer)
    #     cv2.waitKey(0)
    # print(stacked.shape)
    return cv2.resize(all_stacked, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)

def getInputs(day, spec=ht.sample.SampleSpec):
    layers = [day.layers[name] for name in spec.layers]
    stacked = np.dstack(layers)
    np.nan_to_num(stacked, copy=False)
    # for i in range(spec.numLayers):
    #     layer = stacked[:,:,i]
    #     print(spec.layers[i],layer.dtype, layer.shape, layer.min(), layer.max(), layer.mean(), layer.std())
    #     if layer.dtype == np.uint8:
    #         cv2.imshow('perim', layer*255)
    #     else:
    #         cv2.imshow(spec.layers[i],layer)
    #     cv2.waitKey(0)
    # print(stacked.shape)
    return cv2.resize(stacked, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)

def getOutput(day):
    smaller = cv2.resize(day.layers['ending_perim'], None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    np.nan_to_num(smaller, copy=False)
    # make this explicitly one channel
    return np.expand_dims(smaller, axis=2)

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

def make_generator(preprocessor, days, augmentor=None, use_weather=False):
    def generator():
        while True:
            for day in days:
                if augmentor:
                    day = augmentor.augment(day)
                normed = preprocessor.process(day)
                if use_weather:
                    inp = np.expand_dims(getInputsWithWeather(normed), axis=0)
                else:
                    inp = np.expand_dims(getInputs(normed), axis=0)
                out = np.expand_dims(getOutput(normed), axis=0)
                # assert np.isnan(inp).any() == False
                # assert np.isnan(out).any() == False
                yield inp, out
    return generator

def make_multiprocess_generator(preprocessor, days, augmentor=None):
    gen = make_generator(preprocessor, days, augmentor)

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
    m.fit_generator(generator(), steps_per_epoch=16, epochs=256)
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
