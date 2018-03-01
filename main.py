
import hottopic as ht

def useGui():
    ht.gui.makeApp()

def predict():
    m = ht.model.FireModel()
    avail = ht.util.availableDays()
    oneDay = ht.rawdata.getDay(*avail[0])
    m.predictDay(oneDay)

# if __name__ == '__main__':
#     useGui()
    # predict()


import numpy as np
def sample():
    return np.arange()
