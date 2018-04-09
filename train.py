
import random

import hottopic as ht
import keras
import numpy as np
import cv2

import matplotlib.pyplot as plt

testingBurnNames = ['coldSprings', 'riceRidge']
VULNERABLE_RADIUS = 16

def trainingDataset():
    allBurnNames = list(ht.util.availableBurnNames())
    #peekaboo, ecklund, redDirt2, redDirt, beaverCreek, riceRidge, junkins, gutzler, coldSprings, pineTree, haydenPass
    trainingBurnNames = [b for b in allBurnNames if b not in testingBurnNames]
    trainingDays = []
    for bn in trainingBurnNames:
        trainingDays.extend(ht.rawdata.getAllDays(bn))
    return trainingDays

def testingDataset():
    days = []
    for bn in testingBurnNames:
        days.extend(ht.rawdata.getAllDays(bn))
    return days

def pos_and_neg_vulnerable_pixels(starting_perim, ending_perim, radius=VULNERABLE_RADIUS):
    starting_perim = starting_perim.astype(np.uint8)
    kernel = np.ones((3,3))
    its = int(round((2*(radius)**2)**.5))
    dilated = cv2.dilate(starting_perim, kernel, iterations=its).astype(np.uint8)
    border = dilated - starting_perim

    pos = np.where(np.logical_and(border, ending_perim                ))
    neg = np.where(np.logical_and(border, np.logical_not(ending_perim)))
    # print('pos:', pos)
    # print('neg:', neg)
    return pos, neg

def extract_samples(inp, out, num_samples):
    layers, weather = inp
    # print(layers.shape)
    # print(num_samples)
    # print(out.shape, out.dtype, out)
    starting_perim = layers[:,:,0]
    ending_perim = out
    (posy, posx), (negy, negx) = pos_and_neg_vulnerable_pixels(starting_perim, ending_perim)
    # posy, posx = np.where(out)
    # negy, negx = np.where(out==0)
    if num_samples =='all':
        pos_idx = np.arange(len(posy))
        neg_idx = np.arange(len(negy))
    else:
        assert num_samples % 2 == 0
        pos_idx = np.random.choice(len(posy), min(len(posy), num_samples//2), replace=False)
        neg_idx = np.random.choice(len(negy), min(len(negy), num_samples//2), replace=False)
    posy, posx = posy[pos_idx], posx[pos_idx]
    negy, negx = negy[neg_idx], negx[neg_idx]
    idxs = np.concatenate((posy, negy)), np.concatenate((posx, negx))
    # idxs = np.vstack((ys,xs)).T
    # # print('idxs:', idxs)
    # np.random.shuffle(idxs)
    # idxs = tuple(idxs.T)
    r = ht.sample.SampleSpec.AOIRadius
    padded = np.lib.pad(layers, ((r,r),(r,r),(0,0)), 'constant')
    inps = []
    for oldy, oldx in zip(*idxs):
        # with the padded the indices are off
        y = oldy+r
        x = oldx+r
        window = padded[y-r:y+r+1, x-r:x+r+1]
        metrics = np.array(weather)
        inps.append( [window, metrics] )
    outs = out[idxs]

    pairs = list(zip(inps, outs))
    # print('one pair:')
    # print(pairs[0])
    return pairs, idxs

def samples_to_inps_and_outs(samples):
    ins, outs = zip(*samples)
    layers, weathers = zip(*ins)
    return [np.stack(layers, axis=0), np.stack(weathers, axis=0)], np.stack(outs,axis=0)

def make_generator(days, normalizer, augmentor=None, use_weather=False, batch_size=16):
    pp = ht.preprocess.PreProcessor
    def generator():
        while True:
            samples = []
            for day in days:
                if augmentor:
                    day = augmentor.augment(day)

                # ep = day.layers['ending_perim']
                inp = pp.getInput(day, use_weather)
                out = pp.getOutput(day)
                # print('bad pixels of out')
                # print(out[np.logical_and(out!=1, out!=0)])
                #TODO extract from normed
                pairs, idxs = extract_samples(inp, out, 16)
                samples.extend(pairs)
                # yield np.expand_dims(normed, axis=0), np.expand_dims(out, axis=0)
            random.shuffle(samples)
            # print('samples are')
            # print(samples)
            # input()
            for i in range(0, len(samples), batch_size):
                pairs = samples[i:i+batch_size]
                inp_pair, out = samples_to_inps_and_outs(pairs)
                # print('yielding result')
                # print(result)
                normed = normalizer.normalize(inp_pair)
                yield normed, out

    return generator

def fitPP():
    pp = ht.preprocess.PreProcessor
    auger = ht.augment.Augmentor()

    layer_data = []
    weather_data = []
    for r in range(10):
        print('round', r, 'of 10')
        for day in trainingDataset():
            print('augmenting day', day)
            day = auger.augment(day)
            # ep = day.layers['ending_perim']
            inp = pp.getInput(day, use_weather=True)
            out = pp.getOutput(day)
            pairs, idxs = extract_samples(inp, out, 8)
            for inp_pair, output in pairs:
                layer_inp, weather_inp = inp_pair
                # print(layer_inp.shape)
                # print(weather_inp.shape)
                layer_data.append(layer_inp)
                weather_data.append(weather_inp)

    layer_data = np.array(layer_data)
    weather_data = np.array(weather_data)
    inp = [layer_data, weather_data]

    normer = ht.normalizer.Normalizer()
    print('fitting normalizer')
    normer.fit(inp)
    normer.save('back2patches')

    # ccheck to make sure everything is normalized as expected
    normed = normer.normalize(inp)
    ib, wb = normed
    for c in range(ib.shape[-1]):
        m = np.mean(ib[:,:,:,c])
        print(m)
        std = np.std(ib[:,:,:,c])
        print(std)

    for c in range(wb.shape[-1]):
        m = np.mean(wb[...,c])
        print(m)
        std = np.std(wb[...,c])
        print(std)

def fit_model():
    auger = ht.augment.Augmentor()
    normer = ht.normalizer.Normalizer.fromFile('back2patches')
    days = trainingDataset()
    gen = make_generator(days, normer, auger, use_weather=True)

    m = ht.model.FireModel()
    m.fit_generator(gen(), steps_per_epoch=64, epochs=64)
    m.save('back2patches2')
    # for a in gen():
    #     x=9

def test():
    auger = ht.augment.Augmentor()
    normer = ht.normalizer.Normalizer.fromFile('back2patches')
    days = trainingDataset()

    m = ht.model.load('back2patches2')
    pp = ht.preprocess.PreProcessor
    for day in days:
        day = auger.augment(day)

        # ep = day.layers['ending_perim']
        inp = pp.getInput(day, use_weather=True)
        out = pp.getOutput(day)
        # print('bad pixels of out')
        # print(out[np.logical_and(out!=1, out!=0)])
        pairs, idxs = extract_samples(inp, out, 'all')
        ins, outs = samples_to_inps_and_outs(pairs)
        normed = normer.normalize(ins)
        preds = np.squeeze(m.predict(normed))

        canvas = np.empty_like(out)
        canvas[:,:] = np.nan
        canvas[idxs] = preds
        plt.imshow(canvas)
        plt.show()

        # cv2.imshow('preds', canvas)
        # cv2.waitKey(0)

if __name__ == '__main__':
    test()
    # fit_model()
