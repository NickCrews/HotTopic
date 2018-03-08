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

def make_model(nchannels):
    kernel_size=(5,5)
    # one smaple at a time, dont know the H and W, one channel
    inp_shape = (None,None,nchannels)

    m = Sequential()
    m.add(Conv2D(4, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=inp_shape))
    m.add(Conv2D(8, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    m.add(Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    # m.add(Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    m.summary()
    return m

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
    # make this explicityl one channel
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

def train(epochs=10):
    m = make_model(ht.sample.SampleSpec.numLayers)
    train, test = trainAndTestSets()
    pre = ht.preprocess.PreProcessor.fromFile('fitWithSlope')
    normed = pre.process(train)
    for e in range(epochs):
        for day in normed:
            print(e)
            inp, out = np.expand_dims(getInputs(day),axis=0), np.expand_dims(getOutput(day), axis=0)
            assert np.isnan(inp).any() == False
            assert np.isnan(out).any() == False
            # print(inp.shape, out.shape)
            m.fit(inp, out)
    m.save('models/conv.h5')

def test():
    m = keras.models.load_model('models/conv.h5')
    train, test = trainAndTestSets()
    pre = ht.preprocess.PreProcessor.fromFile('fitWithSlope')
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
        canvas = ht.viz.renderCanvas(day)
        print(out.shape, canvas.shape)
        viz = ht.viz.overlayPredictions(canvas, out)
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
