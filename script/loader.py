

import sys
import pdb

PYTHON_TOOLS = '/data0/datasets/tools'
sys.path.append(PYTHON_TOOLS)

import os
import yaml
from random import shuffle

import mxnet as mx
import numpy as np
import cv2 as cv

import python_tools
import python_tools.data_utils

LIDAR_FILE = 'lidar.yaml'
TRACKLET_FILE = 'tracklet.xml'
HEIGHT_FOLDER = 'height'
DENSITY_FOLDER = 'density'
INTENSITY_FOLDER = 'intensity'

TOPVIEW_RESOLUTION = 0.1
TOPVIEW_DIM        = 300.
TOPVIEW_HALFDIM    = (TOPVIEW_DIM / 2.)

class Loader(mx.io.DataIter):
    def __init__(self, root, batch_size, dims, targets):
        super(Loader, self).__init__(self)
        self.root       = root
        self.batch_size = batch_size
        self.dims       = dims
        self.targets    = (100, targets)

        drives = [drv for drv in os.listdir(self.root)
                    if 'drive' in drv]

        self.samples = []

        for drive in drives:
            lidar_file = os.path.join(drive, LIDAR_FILE)

            lidar_path = os.path.realpath(os.path.join(self.root,
                                                       lidar_file))

            with open(lidar_path, 'r') as stream:
                doc = yaml.load(stream)

            keys_list  = doc.keys()
            drive_list = [drive] * len(keys_list)

            samples = zip(drive_list, keys_list)

            self.samples += samples

        self.provide_data = [('data', (self.batch_size, 
                                       self.dims[0], 
                                       self.dims[1],
                                       self.dims[2]))]

        self.provide_label = [('label', (self.batch_size, 
                                         self.targets[0], 
                                         self.targets[1]))]

        self.reset()

    def reset(self):
        shuffle(self.samples)

        self.index = 0

    def next(self):
        self.data = [mx.nd.zeros((self.batch_size, 
                                  self.dims[0], 
                                  self.dims[1],
                                  self.dims[2]))]
        self.labels = [mx.nd.zeros((self.batch_size, 
                                    self.targets[0], 
                                    self.targets[1]))]

        for idx in range(self.batch_size):
            drive, scan = self.samples[idx]
            tracklet_file = os.path.realpath(os.path.join(self.root, drive, TRACKLET_FILE))
            height_file = os.path.realpath(os.path.join(self.root, drive, HEIGHT_FOLDER, scan))
            density_file = os.path.realpath(os.path.join(self.root, drive, DENSITY_FOLDER, scan))
            intensity_file = os.path.realpath(os.path.join(self.root, drive, INTENSITY_FOLDER, scan))

            tracklet_dict = python_tools.data_utils.get_tracklet_dict(tracklet_file)

            #print tracklet_dict

            height_img    = np.asarray(np.load(height_file),    dtype=np.float32) / 255.
            density_img   = np.asarray(np.load(density_file),   dtype=np.float32) / 255.
            intensity_img = np.asarray(np.load(intensity_file), dtype=np.float32) / 255.

            topview_img   = np.concatenate((height_img, density_img, intensity_img), axis=2)

            topview_img   = np.asarray([np.rollaxis(topview_img, 2)])

            self.data[0][idx,:,:,:] = topview_img.copy()

            key     = int(scan.split('.')[0])
            targets = tracklet_dict.setdefault(key, list())

            labels = []

            for target in targets:
                c = int(target[0])

                target_x = target[1][0,:]
                target_y = target[1][1,:]

                w  = np.ptp(target_x)  / TOPVIEW_RESOLUTION
                xo = TOPVIEW_DIM - (np.mean(target_x) / TOPVIEW_RESOLUTION + TOPVIEW_HALFDIM)
                h  = np.ptp(target_y)  / TOPVIEW_RESOLUTION
                yo = TOPVIEW_DIM - (np.mean(target_y) / TOPVIEW_RESOLUTION + TOPVIEW_HALFDIM)

                tl = (xo-w/2., yo-h/2.)
                br = (xo+w/2., yo+h/2.)

                if (tl[0] < 0) or (tl[1] < 0) or (br[0] >= TOPVIEW_DIM) or (br[1] >= TOPVIEW_DIM):
                    continue

                labels += [[c, xo, yo, w, h]]

            #self.labels[0] += labels

        if self.index > len(self.samples):
            return StopIteration

        batch = mx.io.DataBatch(data=self.data,
                                label=self.labels,
                                pad=self.getpad(),
                                index=self.getindex(),
                                provide_data=self.provide_data,
                                provide_label=self.provide_label)

        print 'yooooooh, returning batch'

        return batch

    def iter_next(self):
        self.samples = self.samples[self.batch_size:] + \
                       self.samples[:self.batch_size]

        self.index += self.batch_size

    def getdata(self):
        return self.data

    def getlabel(self):
        return self.labels

    def getindex(self):
        return range(self.index, self.index+self.batch_size)

    def getpad(self):
        return len(self.samples) - self.index
