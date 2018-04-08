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

def smart_entropy_loss(yhat, y):
    # print(yhat)
    # print(y)
    dif = (y-yhat)
    false_pos = K.exp(dif)
    false_neg = K.exp(-dif)

    bias = K.mean(yhat)-.5
    is_0_heavy = K.exp(-bias)
    is_1_heavy = K.exp(bias)

    bce = K.binary_crossentropy(yhat, y)
    # print(bce)
    weights = (is_0_heavy*false_neg + is_1_heavy*false_pos)/2
    # print(weights)
    total_of_weights = K.sum(weights, keepdims=True)
    # print(total_of_weights)
    result = K.mean( (weights*bce)/total_of_weights, axis=-1 )
    # print(result)
    return result

def my_binary_crossentropy(y_true, y_pred):
    print(y_true, y_pred)
    bce = K.binary_crossentropy(y_true, y_pred)
    print(bce)
    result = K.mean(bce, axis=-1)
    print(result)
    return result

def smart_entropy_metric(y_true, y_pred):
    return K.mean(smart_entropy_loss(y_true, y_pred))

def make_model(sample_spec):
    kernel_size=(5,5)
    # one smaple at a time, dont know the H and W, one channel
    inp_shape = (None,None,sample_spec.numLayers)

    m = Sequential()
    m.add(Conv2D(4, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=inp_shape))
    m.add(Conv2D(8, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    m.add(Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    # m.add(Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.compile(optimizer='rmsprop', loss=smart_entropy_loss, metrics=['accuracy', smart_entropy_metric])
    m.summary()
    return m

def make_model_with_weather(num_channels):
    kernel_size=(5,5)
    # one smaple at a time, dont know the H and W, one channel
    inp_shape = (None,None,num_channels)

    m = Sequential()
    m.add(Conv2D(16, (1,1), strides=(1,1), padding='same', data_format='channels_last', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=inp_shape))

    m.add(Conv2D(16, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.add(Conv2D(16, (1,1), strides=(1,1), padding='same', data_format='channels_last', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.add(Conv2D(32, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.add(Conv2D(32, (1,1), strides=(1,1), padding='same', data_format='channels_last', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.add(Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
    # m.add(Conv2D(1, kernel_size, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=2, activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

    m.compile(optimizer='rmsprop', loss=smart_entropy_loss, metrics=['accuracy', smart_entropy_metric])
    m.summary()
    return m

aug = ht.augment.Augmentor()
def augment(day):
    return au.augment(day)

def make_generator(days, normalizer, augmentor=None, use_weather=False):
    pp = ht.preprocess.PreProcessor
    def generator():
        while True:
            for day in days:
                if augmentor:
                    day = augmentor.augment(day)
                inp, crop_bounds = pp.getInput(day, use_weather)
                out = pp.getOutput(day)
                # for i in range(inp.shape[-1]):
                #     print(i, inp[:,:,i].mean())
                # assert np.isnan(inp).any() == False
                # assert np.isnan(out).any() == False
                normed = normalizer.normalize(inp)
                # print('about to yield', np.expand_dims(normed, axis=0).shape, np.expand_dims(out, axis=0).shape)
                yield np.expand_dims(normed, axis=0), np.expand_dims(out, axis=0)
    return generator

def fitPreprocessorWithAugmentedAndWeather(days):
    # days = days[:3]
    aug = ht.augment.Augmentor()
    augmented = []
    for i in range(10):
        print("augmenting round", i)
        augmented.extend(aug.augment(d) for d in days)

    inputs = [ht.preprocess.PreProcessor.getInput(aug) for aug in augmented]

    normer = ht.normalizer.Normalizer()
    normer.fit(inputs)
    normer.save('normerApril5.json')

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
    aug = ht.augment.Augmentor()
    normer = ht.normalizer.Normalizer.fromFile('normerApril7.json')
    gen = make_generator(days, normer, aug, use_weather=True)
    # for inp, out in gen():
    #     for c in range(inp.shape[-1]):
    #         cv2.imshow('input'+str(c), inp[0,:,:,c])
    #         print("channel mean and std", c, np.mean(inp[0,:,:,c]), np.std(inp[0,:,:,c]))
    #     cv2.imshow('out', out[0,:,:,0])
    #     cv2.waitKey(0)

    trainWithWeather(gen, 'models/convApril7.h5')

def testOn(days):
    pre = ht.preprocess.PreProcessor.fromFile('fitWithAugmented.json')
    normer = ht.normalizer.Normalizer.fromFile('normerApril5.json')

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
    augmentor = ht.augment.Augmentor()
    # auged = [augmentor.augment(day) for day in days]
    pp = ht.preprocess.PreProcessor
    normer = ht.normalizer.Normalizer.fromFile('normerApril5.json')
    # normed = pre.process(auged)
    # m = keras.models.load_model('models/convWithAugmentedBig.h5')
    m = keras.models.load_model('models/convApril7.h5',custom_objects={'smart_entropy_loss': smart_entropy_loss})

    for day in days:
        day = augmentor.augment(day)
        inp, crop_bounds = pp.getInput(day, use_weather=True)
        normed = normer.normalize(inp)
        normed = np.expand_dims(normed, axis=0)
        out = m.predict(normed)
        out = np.squeeze(out)

        real_out = pp.getOutput(day)
        pos = np.count_nonzero(real_out)
        total = real_out.size
        neg = total-pos
        print(pos, total, pos/total, pos/neg)

        out_final = pp.reconstruct_output(out, day)

        canvas = ht.viz.render.renderCanvas(day)
        viz = ht.viz.render.overlayPredictions(canvas, out_final)
        max_dim = max(viz.shape[:2])
        if max_dim > 400:
            viz = cv2.resize(viz, None, fx=400/max_dim, fy=500/max_dim)
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
    m = make_model_with_weather(len(ht.preprocess.PreProcessor.ALL_INPUTS))
    m.fit_generator(generator(), steps_per_epoch=16, epochs=128)
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
