import argparse,logging,os
import mxnet as mx
from symbol.symbol_resnet_ssd import resnet_ssd

import pdb

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(description="command for training ResNet-SDD")
    
    parser.add_argument('--gpus',           type=str,   default='0', 
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir',       type=str,   default='./data/kitti/',
                        help='the input data directory')
    parser.add_argument('--data-type',      type=str,   default='expanded', 
                        help='the dataset type')
    parser.add_argument('--list-dir',       type=str,   default='./',
                        help='the directory which contain the training list file')
    parser.add_argument('--lr',             type=float, default=0.1, 
                        help='initialization learning reate')
    parser.add_argument('--mom',            type=float, default=0.9, 
                        help='momentum for sgd')
    parser.add_argument('--bn-mom',         type=float, default=0.9, 
                        help='momentum for batch normlization')
    parser.add_argument('--wd',             type=float, default=0.0001, 
                        help='weight decay for sgd')
    parser.add_argument('--batch-size',     type=int,   default=256, 
                        help='the batch size')
    parser.add_argument('--workspace',      type=int,   default=512, 
                        help='memory space size(MB) used in convolution, if xpu '
                             ' memory is oom, then you can try smaller vale, such as --workspace 256')
    parser.add_argument('--depth',          type=int,   default=60, 
                        help='the depth of resnet')
    parser.add_argument('--type',          type=str,   default='large_flat', 
                        choices=['[small\', \'medium\', \'large\']_[flat\', \'belly\', \'ramp\']'],
                        help='the type of resnet')
    parser.add_argument('--num-classes',    type=int,   default=1000, 
                        help='the class number of your task')
    parser.add_argument('--aug-level',      type=int,   default=2,      choices=[1, 2, 3],
                        help='level 1: use only random crop and random mirror\n'
                             'level 2: add scale/aspect/hsv augmentation based on level 1\n'
                             'level 3: add rotation/shear augmentation based on level 2')
    parser.add_argument('--num-examples',   type=int,   default=100000, 
                        help='the number of training examples')
    parser.add_argument('--kv-store',       type=str,   default='device', 
                        help='the kvstore type')
    parser.add_argument('--load-epoch',     type=int,   default=0,
                        help='load the model on an epoch using the load-prefix')
    parser.add_argument('--run-epochs',     type=int,   default=50,
                        help='run the model for run-epochs epochs')
    parser.add_argument('--frequent',       type=int,   default=50, 
                        help='frequency of logging')
    parser.add_argument('--memonger',                   default=False,  action='store_true',
                        help='true means using memonger to save momory, https://github.com/dmlc/mxnet-memonger')
    parser.add_argument('--retrain',                    default=False,  action='store_true', 
                        help='true means continue training')
    
    args = parser.parse_args()

    return args

def multi_factor_scheduler(begin_epoch, epoch_size, step=[60, 75, 90], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]

    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) \
                        if len(step_) else None
    
    return lr_scheduler

def main():
    filter_list = [ 16,  64, 128,  256] if 'small'  in args.type else \
                  [ 64, 128, 256,  512] if 'medium' in args.type else \
                  [128, 256, 512, 1024] if 'large'  in args.type else None

    unit_list   = [3, 3, 3] if 'flat'  in args.type else \
                  [3, 4, 3] if 'belly' in args.type else \
                  [3, 4, 5] if 'ramp'  in args.type else None

    bottle_neck = True

    channels    = 32 if 'expanded' in args.data_type else \
                   3 if 'compact'  in args.data_type else None

    dimensions  = 300

    args.aug_level   = 1
    args.num_classes = 10

    symbol = resnet_ssd(units=unit_list, num_stage=len(unit_list), filter_list=filter_list, 
                        num_classes=args.num_classes, data_type=args.data_type, 
                        bottle_neck=bottle_neck, 
                        bn_mom=args.bn_mom, workspace=args.workspace, memonger=args.memonger,
                        nms_thresh=0.5, force_suppress=False, nms_topk=400)

   
    epoch_size = max(int(args.num_examples / args.batch_size), 1)
    begin_epoch = args.load_epoch if args.load_epoch else 0
    
    if not os.path.exists("./model"):
        os.mkdir("./model")
    model_prefix = "model/resnet-{}-{}".format(args.type, args.data_type)
    
    if args.memonger:
        import memonger
        symbol = memonger.search_plan(symbol, data=(args.batch_size, channels, dimensions, dimensions))

    symbol.save(model_prefix+'.json')

    '''
    train = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "train.rec") if args.data_type == 'cifar10' else
                              os.path.join(args.data_dir, "train_256_q90.rec") if args.aug_level == 1
                              else os.path.join(args.data_dir, "train_480_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3, 32, 32) if args.data_type=="cifar10" else (3, 224, 224),
        batch_size          = args.batch_size,
        pad                 = 4 if args.data_type == "cifar10" else 0,
        fill_value          = 127,  # only used when pad is valid
        rand_crop           = True,
        max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
        min_random_scale    = 1.0 if args.data_type == "cifar10" else 1.0 if args.aug_level == 1 else 0.533,  # 256.0/480.0
        max_aspect_ratio    = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 0.25,
        random_h            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 36,  # 0.4*90
        random_s            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        random_l            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        max_rotate_angle    = 0 if args.aug_level <= 2 else 10,
        max_shear_ratio     = 0 if args.aug_level <= 2 else 0.1,
        rand_mirror         = True,
        shuffle             = True,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)
    '''

    '''
    val = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "val.rec") if args.data_type == 'cifar10' else
                              os.path.join(args.data_dir, "val_256_q90.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = (3, 32, 32) if args.data_type=="cifar10" else (3, 224, 224),
        rand_crop           = False,
        rand_mirror         = False,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)
    '''

    ''' 
    model = mx.module.Module(
        context             = devs,
        symbol              = symbol,
        data_names          = ['data'],
        label_names         = ['label'],
        )
    '''

    ''''
    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = ['acc'],
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(args.batch_size, args.frequent),
        epoch_end_callback = checkpoint)
    '''

    '''
    initializer = mx.initializer.MSRAPrelu(factor_type='avg', slope=0.)
    model.init_params(initializer=initializer, arg_params=None, aux_params=None, allow_missing=False, force_init=False)

    model.save_checkpoint(model_prefix, begin_epoch, save_optimizer_states=False)
    '''

    # logging.info("top-1 and top-5 acc is {}".format(model.score(X = val,
    #               eval_metric = ['acc', mx.metric.create('top_k_accuracy', top_k = 5)])))

if __name__ == "__main__":
    args = parse_arguments()
    
    logging.info(args)
    
    main()
