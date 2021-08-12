from psana import *
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import cv2

############################
# functions:               #
#     - show_image         #
#     - clip_image         #
#     - flip_image         #
#     - add_noise_image    #
#     - rotate_img         #
#     - rand_crop          #
#     - store_data         #
#     - loop_dir_to_array  #
############################


def show_image(idx, datasource_dir):
    """
    idx - the number of event in a run
    datasource_dir - directory of xtc file
    return - a numpy array image
    exaple usage: img = show_image(8, 'e001-r0084-s80-c00.xtc')
    """
    directory = datasource_dir.split('/')[0]
    ds = DataSource(datasource_dir)
    det = Detector(DetNames()[0][1], ds.env())
    iterator = ds.events()
    for i, evt in enumerate(ds.events()):
        if i == idx:
            print("fetching event number {} from {}".format(idx, directory))
            img = det.image(evt)
            plt.imshow(img)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("referenced image")
            plt.show()
            return img
        elif i < idx:
            continue
        else:
            break


def clip_image(img, show=False):
    """
    clip image to square shape 
    img - numpy array
    return: clipped image.
    example: Original image size is 512 x 648, after clip is 512 x 512
    """
    h, w = img.shape
    if h == w:
        return img
    elif h < w:
        clipped_img = img[:, :h]
    else:
        clipped_img = img[:w, :]
    if show:
        plt.imshow(clipped_img)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("clip image to square")
        plt.show()
    return clipped_img


def flip_image(img, axis, show=False):
    """
    img - numpy array
    axis = 0 means flip vertically
    axis = 1 means flip hotizontally
    return flipped image.
    """
    flipped_img = np.flip(img, axis)
    if show:
        plt.imshow(flipped_img)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("flip image")
        plt.show()
    return flipped_img


def add_noise_image(img, mu=0, show=False):
    """
    Default sigma is 1/10 of the max value of intensity.
    Default mu is 0
    """
    sigma = int(np.amax(img) / 10)
    noise = np.random.normal(mu, sigma, size=img.shape)
    noise_img = img + noise
    if show:
        plt.imshow(noise_img)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("add noise to image")
        plt.show()
    return noise_img


def rotate_img(img, k=1, show=False):
    """
    k - the number of clockwise 90 degree the img will rotate
    """
    rot_img = np.rot90(img, k)
    if show:
        plt.imshow(rot_img)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("rotate image")
        plt.show()
    return rot_img


def rand_crop(img, show=False):
    """
    img: numpy array
    return: a numpy img with random crop from crop
    """
    w, h = img.shape
    crop_size = (int(w * 0.8), int(h * 0.8))
    x, y = np.random.randint(h - crop_size[0]), np.random.randint(w - crop_size[1])
    img = img[y:y+crop_size[0], x:x+crop_size[1]]
    if show:
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("random cropping")
        plt.imshow(img)
        plt.show()
    return img

def store_data(data, label, f):
    """
    f - a h5py file. e.g. f = h5py.File(<file_dir>, "a")
    data - a list like object that we want to put into data dataset of f
    lable - a list of labels. clipped_edge: 0, airy_ring: 1; hot_spot: 2
    enlarge size of f, then store images and labels into it
    """
    assert(len(data) > 0)
    img_dset = f['image']
    label_dset = f['label']
    start_i = len(img_dset)
    end_i = start_i + len(data)
    h = len(data[0])
    w = len(data[0][0])
    img_dset.resize((end_i, h, w))
    label_dset.resize((end_i,))
    img_dset[start_i:] = data[:]
    label_dset[start_i:] = label[:]


def loop_dir_to_array(directory, data):
    """
    will loop through directory and store all events data into data
    data - a list
    return - data
    """ 
    def evt2array(xtc_file, data):
        ds = DataSource(xtc_file)
        det = Detector(DetNames()[0][1], ds.env())
        for nevt, evt in enumerate(ds.events()):
            img = det.image(evt)
            img = clip_image(img)
            data.append(img.tolist())
        return data
    for filename in os.listdir(directory):
        joined_dir = os.path.join(directory, filename)
        data = evt2array(joined_dir, data)
    return data

