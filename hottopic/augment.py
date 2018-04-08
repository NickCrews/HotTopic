
'''augment.py
Much of this is taken from https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py'''

from time import localtime, strftime
from math import ceil
import multiprocessing
# import threading

import numpy as np
import cv2
import scipy.ndimage as ndi

import hottopic as ht

def transform(args):
    channel, final_affine_matrix, final_offset = args
    return ndi.interpolation.affine_transform(
        channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode='constant',
        cval=0.)

from multiprocessing.pool import ThreadPool
_pool = ThreadPool(8)

def apply_transform(x,
                    transform_matrix,
                    fill_mode='constant',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    channel_axis = 2
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    inputs = [(x_channel, final_affine_matrix, final_offset) for x_channel in x]
    transformed = None
    # tp = multiprocessing.pool.ThreadPool(8)

    try:
        transformed = _pool.map(transform, inputs)
        # pool.close()
        # pool.join()
    except RuntimeError:
        # somethign failed,try again
        transformed = [transform(inp) for inp in inputs]

    x = np.stack(transformed, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def transform_matrix_offset_center(matrix, x, y):
    '''Given a 3X3 affine transform matrix and the width and height of an image,
    make it so the transform happens around the center of the image, not the
    UL corner'''
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def flip_axis(x, axis):
    #axis = 0 means that we don't change the axes,
    #which means we iterate through the rows backwards
    #which means flipping vertically, along a horizontal axis
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

class Augmentor(object):

    def __init__(self,
                rotation_range=180.,
                # shear_range=0.,
                zoom_range=.1,
                # fill_mode='nearest',
                # cval=0.,
                horizontal_flip=True,
                vertical_flip=True
                # rescale=None
                ):
        self.rotation_range = rotation_range
        # self.width_shift_range = width_shift_range
        # self.height_shift_range = height_shift_range
        # self.brightness_range = brightness_range
        # self.shear_range = shear_range
        self.zoom_range = zoom_range
        # self.fill_mode = fill_mode
        # self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        # self.rescale = rescale

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

    def augment(self, day):
        names = list(day.layers.keys())
        layers = [day.layers[name] for name in names]
        img = np.dstack(layers)

        # apply the transforms
        params = self.generate_parameters()
        theta, zx, zy, flip_hor, flip_ver = params
        print('augmenting {} with'.format(day.burn.name+day.date), params)

        # if there is something in the corner of the image, it will be cut off with a rotation
        h,w = img.shape[:2]
        rh, rw = h/2, w/2
        center2corner = ( rh**2 + rw**2 )**.5 #distance from center to corner
        r = max(center2corner-rh, center2corner-rw) #needed padding
        r *= max(zx,zy)
        r = int(ceil(r))
        padded = np.lib.pad(img, ((r,r),(r,r),(0,0)), 'constant')
        new_img = self.transform_layers(padded, params)
        new_weather = self.transform_weather(day.weather, params)

        # cv2.imshow('padded',padded[:,:,0])
        # cv2.imshow('img',img[:,:,0])
        # cv2.imshow('newimg',new_img[:,:,0])
        # cv2.waitKey(0)

        # make a new Burn object
        burnName = day.burn.name + '_augmented_' + strftime("%d%b%H_%M", localtime())
        burn_layers = {name: new_img[:,:,names.index(name)] for name in ht.rawdata.Burn.LAYERS}
        new_burn = ht.rawdata.Burn(burnName, burn_layers)
        #make a new day object
        starting_perim = new_img[:,:, names.index('starting_perim')]
        ending_perim = new_img[:,:, names.index('ending_perim')]
        new_day = ht.rawdata.Day(new_burn, day.date, new_weather, starting_perim, ending_perim)

        # make the Burn object remember the Day object
        new_burn.days[new_day.date] = new_day

        return new_day

    def flow(self, days):
        while True:
            for d in days:
                yield self.augment(d)

    def generate_parameters(self):
        if self.rotation_range:
            theta = np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        # if self.shear_range:
        #     shear = np.deg2rad(np.random.uniform(-self.shear_range, self.shear_range))
        # else:
        #     shear = 0

        flip_hor = False if not self.horizontal_flip else np.random.random() < 0.5
        flip_ver = False if not self.vertical_flip else np.random.random() < 0.5

        return theta, zx, zy, flip_hor, flip_ver

    def transform_layers(self, img, params):
        theta, zx, zy, flip_hor, flip_ver = params

        transform_matrix = None
        if theta != 0:
            rad = theta * np.pi/180
            rotation_matrix = np.array([[np.cos(rad), -np.sin(rad), 0],
                                        [np.sin(rad), np.cos(rad), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        # if shear != 0:
        #     shear_matrix = np.array([[1, -np.sin(shear), 0],
        #                             [0, np.cos(shear), 0],
        #                             [0, 0, 1]])
        #     transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = img.shape[:2]
            # make the transform happena round the center of the image, not the UL corner
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            img = apply_transform(img,transform_matrix, fill_mode='constant', cval=np.nan)

        if flip_hor:
            # LR is reversed
            img = flip_axis(img,1)

        if flip_ver:
            # up down are reversed
            img = flip_axis(img,0)

        return img

    def transform_weather(self, weather, params):
        temp, dewpt, temp2, wdir, wspeed, precip, hum = weather
        theta, zx, zy, flip_hor, flip_ver = params

        # Theta is measured CCW, but our wind directions are CW
        wdir = (wdir-theta) % 360

        # fix the zooming. turn polar wind into NS and EW components, scale, then convert back to polar
        # careful, our axes are pointing north, and CW is positiv direction not CCW
        in_radians = wdir*np.pi/180
        NS = wspeed * np.cos(in_radians) #0-> max, pi/2->0
        EW = wspeed * np.sin(in_radians) #0-> 0, pi/2->max
        NS *= zy
        EW *= zx
        wspeed = np.sqrt(NS**2 + EW**2)
        wdir = (np.arctan2(EW, NS)*180/np.pi) % 360

        if flip_hor:
            wdir = (-wdir) % 360
        if flip_ver:
            wdir = (180-wdir) % 360

        weather = temp, dewpt, temp2, wdir, wspeed, precip, hum
        return weather
