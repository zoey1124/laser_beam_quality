import numpy as np
from psana import * 
import matplotlib.pyplot as plt
from scipy import signal, ndimage


def average_img(datasource_dir):
    """
    datasource_dir: directory of xtc format file 
    return: numpy array that contains the average image of all events in the run
    """
    ds = DataSource(datasource_dir)
    det = Detector(DetNames()[0][1], ds.env())
    iterator = ds.events()
    img = None
    num_evt = 0
    for i, evt in enumerate(ds.events()):
        num_evt += 1
        if i == 0:
            img = det.image(evt)
        else:
            img += det.image(evt)
    return img / num_evt

def xcut(img):
    """
    Show the xcut image and return xcut img.
    """
    im_cent = ndimage.measurements.center_of_mass(img)
    xcut = img[int(im_cent[1])]
    plt.plot(xcut)
    plt.show()
    return xcut

def ycut(img, show=True):
    im_cent = ndimage.measurements.center_of_mass(img)
    ycut = [line[int(im_cent[0])] for line in img]
    if show:
        plt.plot(ycut)
        plt.show()
    return ycut
