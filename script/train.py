
import mxnet as mx
import numpy as np

from metric import MultiBoxMetric
from loader import Loader

import argparse
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(description="command for training ResNet-SDD")
    
    parser.add_argument('--data-type',      type=str,   default='expanded', 
                        help='the dataset type')
    parser.add_argument('--depth',          type=int,   default=50, 
                        help='the depth of resnet')
    parser.add_argument('--type',           type=str,   default='large_flat', 
                        #choices=['[small\', \'medium\', \'large\']_[flat\', \'belly\', \'ramp\']'],
                        help='the type of resnet')
    parser.add_argument('--load-epoch',     type=int,   default=0,
                        help='load the model on an epoch load-epoch')
    parser.add_argument('--num-epochs',     type=int,   default=90,
                        help='the number of epochs to run')
    parser.add_argument('--memonger',                   default=False,  action='store_true',
                        help='true means using memonger to save memory, https://github.com/dmlc/mxnet-memonger')
    
    args = parser.parse_args()

    return args

def main(args):

    root = '/data0/datasets/KITTI/dataset/'

    logging.info('Loading data...')
    #--- LOADING DATA --------------------------------------------------------#
    data_chans   = 8 if 'expanded' in args.data_type else \
                   3 if 'compact'  in args.data_type else None
    data_dims    = 300
    label_chans  = 100
    label_dims   = 5
    batch_size   = 10
    train_points = 10
    val_points   = 10

    train_data  = np.random.randn(train_points, data_chans,  data_dims, data_dims)
    train_label = np.random.randn(train_points, label_chans, label_dims)
    val_data    = np.random.randn(val_points,   data_chans,  data_dims, data_dims)
    val_label   = np.random.randn(val_points,   label_chans, label_dims)

    '''
    train_iter  = mx.io.NDArrayIter(train_data, train_label, batch_size, 
                            data_name='data', label_name='label', shuffle=True)
    val_iter    = mx.io.NDArrayIter(val_data,   val_label,   batch_size, 
                            data_name='data', label_name='label')
    '''

    train_iter  = Loader(root, batch_size, 
                         (data_chans, data_dims, data_dims),
                         label_dims)

    val_iter    = Loader(root, batch_size, 
                         (data_chans, data_dims, data_dims),
                         label_dims)
    #-------------------------------------------------------------------------#
    
    logging.info('Loading model...')
    #--- LOADING MODEL -------------------------------------------------------#
    model_prefix = "model/resnet-sdd-{}-{}".format(args.type, args.data_type)
    begin_epoch  = args.load_epoch if args.load_epoch else 0
    end_epoch    = args.num_epochs - begin_epoch

    model = mx.module.Module.load(prefix        = model_prefix, 
                                  epoch         = begin_epoch,
                                  data_names    = ['data'], 
                                  label_names   = ['label'],
                                  logger        = logger,
                                  context       = mx.gpu(0))
    model.bind(data_shapes  = train_iter.provide_data, 
               label_shapes = train_iter.provide_label, 
               for_training = True,
               force_rebind = True)
    model.init_optimizer(optimizer  = 'adadelta', 
                         force_init = True)
    #-------------------------------------------------------------------------#
    
    logging.info('training model...')
    #--- TRAINING MODEL ------------------------------------------------------#
    frequent = 10

    model.fit(train_data            = train_iter, 
              eval_data             = val_iter,
              eval_metric           = MultiBoxMetric(),
              batch_end_callback    = mx.callback.Speedometer(train_iter.batch_size, frequent),
              epoch_end_callback    = mx.callback.do_checkpoint(model_prefix, period=10),
              eval_end_callback     = mx.callback.LogValidationMetricsCallback(),
              begin_epoch           = begin_epoch,
              num_epoch             = args.num_epochs) 
    model.save_checkpoint(prefix = model_prefix, 
                          epochs = end_epochs)
    #-------------------------------------------------------------------------#

    accuracy = model.score(val_iter, MultiBoxMetric())[0][1]
    assert accuracy > 0.80, "Low validation accuracy."

if __name__ == "__main__":
    args = parse_arguments()
    
    logging.info(args)
    
    main(args)