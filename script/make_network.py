
import argparse
import logging
import os

import mxnet as mx
import numpy as np

from symbol.symbol_resnet_ssd import resnet_ssd

import pdb

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(description="command for training ResNet-SDD")
    
    parser.add_argument('--data-type',      type=str,   default='expanded', 
                        help='the dataset type')
    parser.add_argument('--workspace',      type=int,   default=512, 
                        help='memory space size(MB) used in convolution, if xpu '
                             ' memory is oom, then you can try smaller vale, such as --workspace 256')
    parser.add_argument('--depth',          type=int,   default=50, 
                        help='the depth of resnet')
    parser.add_argument('--type',          type=str,   default='large_flat', 
                        #choices=['[small\', \'medium\', \'large\']_[flat\', \'belly\', \'ramp\']'],
                        help='the type of resnet')
    parser.add_argument('--num-classes',    type=int,   default=3, 
                        help='the class number of your task')
    parser.add_argument('--aug-level',      type=int,   default=0,      choices=[0, 1, 2, 3],
                        help='level 1: use only random crop and random mirror\n'
                             'level 2: add scale/aspect/hsv augmentation based on level 1\n'
                             'level 3: add rotation/shear augmentation based on level 2')
    parser.add_argument('--bn-mom',         type=float, default=0.9, 
                        help='momentum for batch normlization')
    parser.add_argument('--load-epoch',     type=int,   default=0,
                        help='load the model on an epoch using the load-prefix')
    parser.add_argument('--memonger',                   default=False,  action='store_true',
                        help='true means using memonger to save memory, https://github.com/dmlc/mxnet-memonger')
    
    args = parser.parse_args()

    return args

def main(args):
    filter_list     = [ 16,  64, 128,  256] if 'small'  in args.type else \
                      [ 64, 128, 256,  512] if 'medium' in args.type else \
                      [ 64, 256, 512, 1024] if 'large'  in args.type else \
                      [128, 256, 512, 1024] if 'huge'   in args.type else None
    unit_list       = [3, 3, 3] if 'flat'  in args.type else \
                      [3, 4, 3] if 'belly' in args.type else \
                      [3, 4, 5] if 'ramp'  in args.type else None
    channels        =  32 if 'expanded' in args.data_type else \
                       3 if 'compact'  in args.data_type else None
    dimensions      = 300
    sizes           = [
                       [.01, .0141], 
                       [.02, .0316], 
                       [.05, .0707], 
                       [.10, .1410], 
                       #[.20, .3160], 
                       #[.50, .7070]
                      ]
    ratios          = [
                       1.,  
                       2., 1./2., 
                       3., 1./3.
                      ]
    normalizations  = [ 
                       -1, 
                       -1, 
                       -1, 
                       -1
                      ]
    steps           = [ x / 300.0 for x in [  
                                              8., 
                                             16., 
                                             32., 
                                             64., 
                                            #100., 
                                            #300.
                                           ]
                      ]
    num_channels    = []
    nms_thresh      = 0.5
    force_suppress  = False
    nms_topk        = 400

    symbol = resnet_ssd(units=unit_list, num_stage=len(unit_list), filter_list=filter_list, num_classes=args.num_classes, data_type=args.data_type, 
                        sizes=sizes, ratios=ratios, normalizations=normalizations, steps=steps, num_channels=num_channels,
                        bn_mom=args.bn_mom, workspace=args.workspace, memonger=args.memonger,
                        nms_thresh=nms_thresh, force_suppress=force_suppress, nms_topk=nms_topk)

    begin_epoch = args.load_epoch if args.load_epoch else 0

    model_prefix = "model/resnet-{}".format(args.depth)

    load_symbol, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, begin_epoch)
    
    data = mx.io.NDArrayIter(data=np.zeros((1, channels, dimensions, dimensions)),
                             label=np.zeros((1, 10, 5)), label_name='label')

    model = mx.module.Module(context             = mx.gpu(),
                             symbol              = symbol,
                             data_names          = ['data'],
                             label_names         = ['label'])

    model.bind(data.provide_data, data.provide_label)

    initializer = mx.initializer.MSRAPrelu(factor_type='avg', slope=0.)

    model.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params, allow_missing=True, force_init=True)

    model_prefix = "model/resnet-sdd-{}-{}".format(args.type, args.data_type)

    model.save_checkpoint(model_prefix, begin_epoch, save_optimizer_states=False)

if __name__ == "__main__":
    args = parse_arguments()
    
    logging.info(args)
    
    main(args)
