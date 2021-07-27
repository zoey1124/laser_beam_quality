import numpy as np
from psana import * 
import matplotlib.pyplot as plt
from scipy import signal, ndimage


# use the following function to see the image of the idx event in given datasource
def show_image(idx, datasource_dir):
    directory = datasource_dir.split('/')[0]
    ds = DataSource(datasource_dir)
    det = Detector(DetNames()[0][1], ds.env())
    iterator = ds.events()
    for i, evt in enumerate(ds.events()):
        if i == idx:
            print("fetching event number {} from {}".format(idx, directory))
            img = det.image(evt)
            plt.imshow(img)
            plt.show()
            return img
        elif i < idx:
            continue
        else:
            break


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


def draw_diff_1d(img1, img2, show=True):
    if not type(img1) is np.ndarray:
        img1 = np.array(img1)
    if not type(img2) is np.ndarray:
        img2 = np.array(img2)
    diff = abs(img1 - img2)
    if show:
        plt.plot(diff)
        plt.show()
    return diff
def draw_diff_2d(img1, img2, show=True):
    if not type(img1) is np.ndarray:
        img1 = np.array(img1)
    if not type(img2) is np.ndarray:
        img2 = np.array(img2)
    diff = abs(img1 - img2)
    if show:
        plt.imshow(diff)
        plt.show()
    return diff


def compare_ref(ref, run_dir, threshold):
    """
    Inputs:
    ref - 2d array or np array as the referenced laser beam quality that we want to compare to.
    run_fir - the xtc file directory for all events we want to compare to
    threshold - integer. If difference value exceed threshold, will report aberration
    """
    if not type(ref) is np.ndarray:
        ref = np.array(ref)
    ds = DataSource(run_dir)
    det = Detector(DetNames()[0][1], ds.env())
    iterator = ds.events()
    for i, evt in enumerate(ds.events()):
        img = det.image(evt)
        diff = abs(img - ref)
        if (diff > threshold).any():
            print("In event {}, there is an aberration.".format(i))
            return
    print("No Aberration!")
