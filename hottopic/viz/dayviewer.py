
import sys
import os
import time

import threading
import queue

import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from PyQt5 import QtCore, QtGui, QtWidgets, uic

# dynamically generate the gui skeleton file from the ui file
# with open('hottopic' + os.sep + 'viz' + os.sep + 'basic_dayviewer.py', 'w') as pyfile:
#     uic.compileUi('hottopic' + os.sep + 'viz' + os.sep + 'dayviewer.ui', pyfile)

import hottopic as ht
# from . import basic_dayviewer

class DayViewer(object):

    def __init__(self, day=None):
        self.day=None
        if self.day is not None:
            self.show(self.day)

    def show(self, day):
        layer_num = 0
        layer_name = day.burn.LAYERS[layer_num]
        window_name = "{} - {} - {}".format(day.burn.name, day.date, layer_name)
        cv2.namedWindow(window_name)
        canvas = ht.viz.render.renderCanvas(day, base=ht.rawdata.Burn.LAYERS[layer_num])
        cv2.imshow(window_name, canvas)
        cv2.moveWindow(window_name, 25, 25)
        # ht.viz.render.renderWindRose(day, nsector=16, now=True)
        ht.viz.render.showWindData(day)
        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            keyCode = cv2.waitKey(50)
            if keyCode in map(ord, ['q', 'Q']) or not plt.get_fignums():
                cv2.destroyWindow(window_name)
                plt.close('all')
                break

            need_update = False
            try:
                new_layer_num = int(chr(keyCode))
                if new_layer_num in range(len(ht.rawdata.Burn.LAYERS)) and new_layer_num != layer_num:
                    layer_num = new_layer_num
                    need_update = True
            except:
                pass
            if need_update:
                layer_name = day.burn.LAYERS[layer_num]
                new_window_name = "{} - {} - {}".format(day.burn.name, day.date, layer_name)
                if window_name != new_window_name:
                    cv2.destroyWindow(window_name)
                    window_name = new_window_name
                    canvas = ht.viz.render.renderCanvas(day, base=layer_name)
                    cv2.imshow(window_name, canvas)
                    cv2.moveWindow(window_name, 25, 25)
