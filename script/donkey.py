

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

class Loader(mx.io.DataIter):
    def __init__(self, root, batch_size, dims, targets):
        super(Loader, self).__init__(self)
        self.root       = root
        self.batch_size = batch_size
        self.dims       = dims
        self.targets    = (10, targets)

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

            #height_mat = cv.cv.Load(height_file)

            #print height_mat

            pdb.set_trace()





        if self.index > len(self.samples):
            return StopIteration

        batch = mx.io.DataBatch(data=self.data,
                                label=self.labels,
                                pad=self.getpad(),
                                index=self.getindex(),
                                provide_data=self.provide_data,
                                provide_label=self.provide_label)

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
