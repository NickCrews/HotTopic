from time import localtime, strftime
import os

print('importing keras...')
import keras.models
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Concatenate, Input
from keras.optimizers import SGD, RMSprop
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
print('\rimporting keras...done')

import hottopic as ht

class ImageBranch(Sequential):

    def __init__(self, nchannels, kernelDiam):
        super().__init__()
        # there is also the starting perim which is implicitly gonna be included
        nchannels += 1
        input_shape = (kernelDiam, kernelDiam, nchannels)

        self.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), input_shape=input_shape))
        self.add(Conv2D(32, kernel_size=(3,3), strides=(1,1),
                        activation='sigmoid'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        self.add(Dropout(0.5))
        self.add(Conv2D(64, (3,3), activation='relu'))
        self.add(MaxPooling2D(pool_size=(2,2)))
        self.add(Dropout(0.5))
        self.add(Flatten())
        self.add(Dense(128, activation='relu'))
        self.add(Dropout(0.5))

        self.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            metrics=['accuracy'])

class BaseModel(object):

    DEFAULT_EPOCHS = 1
    DEFAULT_BATCHSIZE = 1000

    def __init__(self, kerasModel=None, preProcessor=None):
        self.kerasModel = kerasModel
        self.pp = preProcessor

    # def fit(self, trainingDataset, validatateDataset=None, epochs=DEFAULT_EPOCHS,batch_size=DEFAULT_BATCHSIZE):
    #     assert self.kerasModel is not None, "You must set the kerasModel within a subclass"
    #     assert self.pp is not None, "You must set the preProcessor within a subclass"
    #
    #     print('training on ', trainingDataset)
    #     # get the actual samples from the collection of points
    #     (tinputs, toutputs), ptList = self.pp.process(trainingDataset)
    #     if validatateDataset is not None:
    #         (vinputs, voutputs), ptList = self.pp.process(validatateDataset)
    #         history = self.kerasModel.fit(tinputs, toutputs, batch_size=batch_size, epochs=epochs, validation_data=(vinputs, voutputs))
    #     else:
    #         history = self.kerasModel.fit(tinputs, toutputs, batch_size=batch_size, epochs=epochs)
    #     return history

    def predictDay(self, day):
        assert self.kerasModel is not None, "You must set the kerasModel within a subclass"
        inp, out = self.pp.processDay(day)
        results = self.kerasModel.predict(inp)
        return results
    #
    # def fit_generator(self, directory, valDirectory=None, epochs=DEFAULT_EPOCHS, steps_per_epoch=1):
    #     assert self.kerasModel is not None, "You must set the kerasModel within a subclass"
    #     assert self.pp is not None, "You must set the preProcessor within a subclass"
    #
    #     gen = preprocess.streamFromDir(directory)
    #     valGen = preprocess.streamFromDir(valDirectory) if valDirectory else None
    #     kwargs = {'steps_per_epoch':steps_per_epoch, 'epochs':epochs, 'validation_data':valGen, 'verbose':1}
    #     history = self.kerasModel.fit_generator(gen, **kwargs)
    #     return history
    #
    # def predictDay(self, day):
    #     self.pp.
    #
    # def predict(self, dataset):
    #     assert self.kerasModel is not None, "You must set the kerasModel within a subclass"
    #     (inputs, outputs), ptList = self.pp.process(dataset)
    #     results = self.kerasModel.predict(inputs).flatten()
    #     resultDict = {pt:pred for (pt, pred) in zip(ptList, results)}
    #     return resultDict

    def save(self, name=None):
        if name is None:
            name = strftime("%d%b%H_%M", localtime())
        if "models"+os.sep not in name:
            name = "models" + os.sep + name
        if not name.endswith(os.sep):
            name += os.sep

        if not os.path.isdir(name):
            os.mkdir(name)

        className = str(self.__class__.__name__)
        with open(name+'class.txt', 'w') as f:
            f.write(className)
        self.kerasModel.save(name+'model.h5')

def load(modelFolder):
    if 'models'+os.sep not in modelFolder:
        modelFolder = 'models' + os.sep + modelFolder
    assert os.path.isdir(modelFolder), "{} is not a folder".format(modelFolder)

    if not modelFolder.endswith(os.sep):
        modelFolder += os.sep

    modelFile = modelFolder + 'model.h5'
    model = keras.models.load_model(modelFile)

    objFile = modelFolder + 'class.txt'
    # print(objFile)
    with open(objFile, 'r') as f:
        classString = f.read().strip()
    # print('classString is ', classString)
    # print(globals())
    class_ = globals()[classString]
    obj = class_(kerasModel=model)
    # print('done! returning', obj)
    return obj

class OurModel(BaseModel):

    def __init__(self, kerasModel=None):
        numWeatherInputs = 8
        usedLayers = ['dem','ndvi', 'aspect', 'band_2', 'band_3', 'band_4', 'band_5'] #, 'slope'
        AOIRadius = 30
        pp = preprocess.PreProcessor(numWeatherInputs, usedLayers, AOIRadius)

        if kerasModel is None:
            kerasModel = OurModel.createModel(pp)

        super().__init__(kerasModel, pp)

    def createModel(pp):
        # make our keras Model
        kernelDiam = 2*pp.AOIRadius+1
        wb = Input((pp.numWeatherInputs,),name='weatherInput')
        ib = ImageBranch(len(pp.whichLayers), kernelDiam)

        concat = Concatenate(name='mergedBranches')([wb,ib.output])
        out = Dense(1, kernel_initializer = 'normal', activation = 'sigmoid',name='output')(concat)
        # print("concat and out info:", concat.shape, out.shape)
        kerasModel = Model([wb, ib.input], out)

        # self.add(Concatenate([self.wb, self.ib]))
        sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
        #rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        kerasModel.compile(loss = 'binary_crossentropy', optimizer = sgd, metrics = ['accuracy'])
        return kerasModel
