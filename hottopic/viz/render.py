from time import localtime, strftime
import csv
import math

import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
# print('Successfully imported pyplot')

import hottopic as ht

def renderBurn(burn):
    dem = burn.layers['dem']
    return util.normalize(dem)

def renderDay(day):
    '''render the start and end perimeters over the dem.

    use HSV color space. value of pixel is DEM. Colored pixels
    represent burned area. color determines if it was burned before or after'''
    v = util.normalize(day.layers['dem'])
    h = np.ones_like(v, dtype=np.float32)*.1
    s = (day.layers['ending_perim']*.8).astype(np.float32)
    h[np.where(day.layers['starting_perim'])] = .3
    hsv = cv2.merge((h,s,v))
    hsv = (hsv*255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def renderUsedPixels(dataset, burnName, date):
    # burnName, date = day.burn.name, day.date
    mask = dataset.masks[burnName][date]
    # bg = day.layers['dem']
    # background = cv2.merge((bg,bg,bg))
    return mask*127

def renderPredictions(dataset, predictions):
    # print('predictions are', predictions.values())
    day2pred = {}
    for pt, pred in predictions.items():
        burnName, date, location = pt
        day = (burnName, date)
        if day not in day2pred:
            day2pred[day] = []
        pair = (location, float(pred))
        # print('storing prediction', pair)
        day2pred[day].append(pair)

    # print('these are all the original days:', day2pred.keys())
    results = {}
    for (burnName, date), locsAndPreds in day2pred.items():
        # print('locs and preds', locsAndPreds)
        locs, preds = zip(*locsAndPreds)
        # print('reds:', preds)
        xs,ys = zip(*locs)
        preds = [pred+1 for pred in preds]
        # print((xs,ys))
        # print(max(preds), min(preds))
        # print(len(xs), len(preds))
        burn = dataset.data.burns[burnName]
        canvas = np.zeros(burn.layerSize, dtype=np.float32)
        # print(canvas)
        canvas[(xs,ys)] = np.array(preds, dtype=np.float32)
        results[(burnName, date)] = canvas
    return results

def createCanvases(dataset):
    result = {}
    for burnName, date in dataset.getUsedBurnNamesAndDates():
        burn = dataset.data.burns[burnName]
        day = dataset.data.getDay(burnName, date)
        result[(burnName, date)] = renderCanvas(day)
    return result

def renderCanvas(day, base='dem', start_perimeter=True, end_perimeter=True):
    '''return the DEM with the start and end perims overlayed, in BGR'''
    normedDEM = ht.util.normalize(day.layers[base])
    canvas = cv2.cvtColor(normedDEM, cv2.COLOR_GRAY2BGR)
    if end_perimeter:
        ep = day.layers['ending_perim'].astype(np.uint8).copy()
        im2, endContour, heirarchy = cv2.findContours(ep, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, endContour, -1, (0,0,1), 1)
    if start_perimeter:
        sp = day.layers['starting_perim'].astype(np.uint8).copy()
        im2, startContour, hierarchy = cv2.findContours(sp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, startContour, -1, (0,1,0), 1)
    return canvas

def overlayPredictions(canvas, predictions):
    clipped = np.clip(predictions,0,1)
    yellowToRed = np.dstack((np.zeros_like(clipped), 1-clipped, np.ones_like(clipped)))
    ytrh, ytrs, ytrv = cv2.split(cv2.cvtColor(yellowToRed, cv2.COLOR_BGR2HSV))
    ch,cs,cv = cv2.split(cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV))
    # copy perim colors onto yellowToRed
    ytrh[cs!=0] = ch[cs!=0]
    return cv2.cvtColor(cv2.merge((ytrh, ytrs, cv)), cv2.COLOR_HSV2BGR)

def overlay(predictionRenders, canvases):
    result = {}
    for burnName, date in sorted(canvases):
        canvas = canvases[(burnName, date)].copy()
        render = predictionRenders[(burnName, date)]
        yellowToRed = np.dstack((np.ones_like(render), 1-(render-1), np.zeros_like(render)))
        canvas[render>1] = yellowToRed[render>1]
        result[(burnName, date)] = canvas
    return result

def visualizePredictions(dataset, predictions):
    # print('these are all the burns Im going to start rendering:', predictions.keys())
    predRenders = renderPredictions(dataset, predictions)
    canvases = createCanvases(dataset)
    overlayed = overlay(predRenders, canvases)
    return overlayed

def showPredictions(predictionsRenders):
    # sort by burn
    # print("Here are all the renders:", predictionsRenders.keys())
    burns = {}
    for (burnName, date), render in predictionsRenders.items():
        if burnName not in burns:
            burns[burnName] = []
        burns[burnName].append((date, render))

    # isRunning = {}
    # print("These are all the burns I'm showing:", burns.keys())
    for burnName, frameList in burns.items():
        frameList.sort()
        fig = plt.figure(burnName, figsize=(8, 6))
        ims = []
        pos = (30,30)
        color = (0,0,1.0)
        size = 1
        thickness = 2
        for date, render in frameList:
            withTitle = render.copy()
            cv2.putText(withTitle,date, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness=thickness)
            im = plt.imshow(withTitle)
            ims.append([im])
        anim = animation.ArtistAnimation(fig, ims, interval=300, blit=True,
                                repeat_delay=0)

        def createMyOnKey(anim):
            def onKey(event):
                if event.key == 'right':
                    anim._step()
                elif event.key == 'left':
                    saved = anim._draw_next_frame
                    def dummy(a,b):
                        pass
                    anim._draw_next_frame = dummy
                    for i in range(len(anim._framedata)-2):
                        anim._step()
                    anim._draw_next_frame = saved
                    anim._step()
                    # print(success)
                    # if not success:
                    #     anim.frame_seq = anim.new_frame_seq()
                    #     anim._step()
                elif event.key =='down':
                    anim.event_source.stop()
                elif event.key =='up':
                    anim.event_source.start()
            return onKey

        # fig.canvas.mpl_connect('button_press_event', onClick)
        fig.canvas.mpl_connect('key_press_event', createMyOnKey(anim))
        plt.show()


def renderWindRose(day, nsector=16, now=True):
    temp, dewpt, temp2, wdir, wspeed, precip, hum = day.weather

    window_name = "{} - {} - Wind Data ({} sectors)".format(day.burn.name, day.date, nsector)
    FIGSIZE_DEFAULT = (8, 8)
    DPI_DEFAULT = 80
    fig = plt.figure(window_name, figsize=FIGSIZE_DEFAULT, dpi=DPI_DEFAULT, facecolor='w', edgecolor='w')
    rect = [0.1, 0.1, 0.8, 0.8]
    ax = ht.viz.windrose.WindroseAxes(fig, rect, facecolor='w', rmax=None)
    fig.add_axes(ax)
    # ax = ht.viz.windrose.WindroseAxes.from_ax()

    ax.bar(wdir, wspeed, nsector=nsector, opening=0.8, edgecolor='white')
    # max_speed = int(wspeed.max())+1
    # bins = np.arange(0, max_speed, 1)
    # ax.contour(wdir, wspeed, bins=bins, cmap=cm.hot)
    ax.set_legend()
    if now:
        plt.show(block=False)
    return ax

def showWindData(day):
    window_name = "{} - {} - Wind Data".format(day.burn.name, day.date)
    fig=plt.figure(window_name)
    ax = fig.add_subplot(111, projection='polar')
    temp, dewpt, temp2, wdir, wspeed, precip, hum = day.weather
    theta = wdir*np.pi/180
    colors = theta
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.scatter(theta, wspeed, c=colors, cmap='hsv', alpha=.75)
    plt.show(block=False)

def renderPerformance(model, samples):
    expCanvas = np.empty_like(samples[0].day.layers['starting_perim'], dtype=np.float32)
    predCanvas = np.empty_like(samples[0].day.layers['starting_perim'], dtype=np.float32)
    expCanvas[:,:] = np.nan
    predCanvas[:,:] = np.nan
    bs = 1024
    numBatches = int(math.ceil(len(samples)//bs))
    for i, batch in enumerate(ht.sample.getBatches(samples, batchSize=bs, shuffle=False)):
        print('predicting on {}/{}!'.format(i, numBatches))
        inp = ht.sample.toModelInput(batch)
        pred = model.predict(inp)
        expected = ht.sample.toModelOutput(batch)
        for s, e, p in zip(batch, expected, pred):
            expCanvas[s.loc] = e
            predCanvas[s.loc] = p

    return expCanvas, predCanvas

def show(*imgs, now=True):
    for i, img in enumerate(imgs):
        plt.figure(i, figsize=(8, 6))
        plt.imshow(img)
    if now:
        plt.show()

def save(img, name):
    fname = 'output/imgs/{}.png'.format(name)
    cv2.imwrite(fname, img)

def renderModel(model):
    kmodel = model.kerasModel
    import os
    from keras.utils import plot_model
    timeString = strftime("%d%b%H:%M", localtime())
    fname = 'output/modelViz/{}.png'.format(timeString)
    try:
        plot_model(kmodel, to_file=fname, show_shapes=True)
        img = cv2.imread(fname)
        return img
    finally:
        if os.path.exists(fname):
            os.remove(fname)

def saveModelDiagram(model):
    from keras.utils import plot_model
    timeString = strftime("%d%b%H:%M", localtime())
    fname = 'output/modelViz/{}.png'.format(timeString)
    plot_model(model, to_file=fname, show_shapes=True)