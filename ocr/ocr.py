
import numpy as np
import os
import urllib
import gzip
import struct
import time

def download_data(url, force_download=True): 
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        urllib.urlretrieve(url, fname)
    return fname

def read_data(label_url, image_url):
    with gzip.open(download_data(label_url)) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_data(image_url), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

path='http://yann.lecun.com/exdb/mnist/'
(train_lbl, train_img) = read_data(
    path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
(val_lbl, val_img) = read_data(
    path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz')

import mxnet as mx

def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

batch_size = 100
train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)

data = mx.symbol.Variable('data')
shape = {"data" : (batch_size, 1, 28, 28)}
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
graph = mx.viz.plot_network(symbol=lenet, shape=shape)
graph.render('ocr', cleanup=True)

epochs = 5
model_prefix = 'LeNet'

class Speedometer(object):
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if (count+1) % (self.frequent) == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                name_value = param.eval_metric.get_name_value()
                param.eval_metric.reset()
                for name, value in name_value:
                    print 'Epoch[{}] Batch [{}]\tSpeed: {:.5} samples/sec\tTrain-{}={:.3}'.format(param.epoch, count+1, speed, name, value)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()

class LogValidationMetricsCallback(object):
    def __call__(self, param):
        if not param.eval_metric:
            return
        name_value = param.eval_metric.get_name_value()
        for name, value in name_value:
            print 'Epoch[{}] Validation-{}={:.3}'.format(param.epoch, name, value)

# create and initialize Module from Symbol
model = mx.module.Module(context=mx.gpu(0), symbol=lenet)
model.bind(train_iter.provide_data, train_iter.provide_label)
model.init_params(mx.initializer.Xavier(rnd_type='gaussian', factor_type='avg', magnitude=2), force_init=True)
model.init_optimizer(optimizer='adadelta', force_init=True)
# save untrained Module
model.save_checkpoint(model_prefix, 0)
# train Module for N epochs
model.fit(train_data=train_iter, eval_data=val_iter,
    	  batch_end_callback = Speedometer(batch_size, 60),
    	  eval_end_callback = LogValidationMetricsCallback(),
	      num_epoch=epochs) 
# save trained Module
model.save_checkpoint(model_prefix, epochs)

accuracy = model.score(val_iter, 'acc')[0][1]
assert accuracy > 0.98, "Low validation accuracy."
