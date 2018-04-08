
import random

import hottopic as ht
import keras
import numpy as np
import cv2

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
    print('pos:', pos)
    print('neg:', neg)
    return pos, neg

def extract_samples(inp, out, num_samples):
    assert num_samples % 2 == 0
    layers, weather = inp
    print(layers.shape)
    print(num_samples)
    print(out.shape, out.dtype, out)
    starting_perim = layers[:,:,0]
    ending_perim = out
    (posy, posx), (negy, negx) = pos_and_neg_vulnerable_pixels(starting_perim, ending_perim)
    # posy, posx = np.where(out)
    # negy, negx = np.where(out==0)
    pos_idx = np.random.choice(len(posy), num_samples//2, replace=False)
    neg_idx = np.random.choice(len(posx), num_samples//2, replace=False)
    posy, posx = posy[pos_idx], posx[pos_idx]
    negy, negx = negy[neg_idx], negx[neg_idx]
    idxs = np.concatenate((posy, negy)), np.concatenate((posx, negx))
    # idxs = np.vstack((ys,xs)).T
    # # print('idxs:', idxs)
    # np.random.shuffle(idxs)
    # idxs = tuple(idxs.T)

    # print('shuffled:', idxs)
    outs = out[idxs]
    print('outs', outs)


def make_generator(days, normalizer, augmentor=None, use_weather=False, batch_size=16):
    pp = ht.preprocess.PreProcessor
    def generator():
        while True:
            samples = []
            for day in days:
                if augmentor:
                    day = augmentor.augment(day)
                inp = pp.getInput(day, use_weather)
                out = pp.getOutput(day)
                # normed = normalizer.normalize(inp)
                #TODO extract from normed
                pairs = extract_samples(inp, out, 16)
                samples.extend(pairs)
                # yield np.expand_dims(normed, axis=0), np.expand_dims(out, axis=0)
            random.shuffle(samples)
            for i in range(0, len(samples), batch_size):
                pairs = samples[i:i+batchSize]
                inps, outs = zip(*pairs)
                # print(inps)
                # print(outs)
                yield np.array(inps), np.array(outs)
    return generator

def main():
    auger = ht.augment.Augmentor()
    normer = ht.normalizer.Normalizer.fromFile('normerApril5')
    days = trainingDataset()
    gen = make_generator(days, normer, auger, use_weather=True)
    for a in gen():
        print(a)



if __name__ == '__main__':
    main()
