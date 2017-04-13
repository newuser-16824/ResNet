
import mxnet as mx

from common import multibox_layer

import pdb

def residual_unit(data, num_filter, stride, dim_match, name, bn_mom=0.9, workspace=512, memonger=False):    
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                  no_bias=True, workspace=workspace, name=name + '_conv1')
    
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                  no_bias=True, workspace=workspace, name=name + '_conv2')
    
    bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
    act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                        workspace=workspace, name=name+'_sc')
    if memonger:
        shortcut._set_attr(mirror_stage='True')
    
    return conv3 + shortcut

def resnet(units, num_stage, filter_list, num_classes, data_type, 
           bn_mom=0.9, workspace=512, memonger=False, num_channels=3):
    num_unit = len(units)
    assert(num_unit == num_stage)

    prefix = '' if num_channels == 3 else 'x'

    data = mx.sym.Variable(name='data')

    body = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name=prefix+'bn_data')
    body = mx.sym.Convolution(data=body, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                              no_bias=True, name=prefix+"conv0", workspace=workspace)
    #body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    #body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    #body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    
    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], (2, 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 workspace=workspace, memonger=memonger)

    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='xbn1')
    body = mx.sym.Activation(data=body, act_type='relu', name='xrelu1')

    return body

def ssd(from_layers, num_classes, sizes, ratios, normalizations, steps, num_channels, 
        nms_thresh=0.5, force_suppress=False, nms_topk=400):

    label = mx.symbol.Variable(name="label")

    loc_preds, cls_preds, anchor_boxes = multibox_layer(from_layers, \
        num_classes, sizes=sizes, ratios=ratios, normalization=normalizations, \
        num_channels=num_channels, clip=False, interm_layer=0, steps=steps)

    tmp = mx.contrib.symbol.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target, \
        ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True, \
        normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_", \
        data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1., \
        normalization='valid', name="loc_loss")

    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    det = mx.contrib.symbol.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes], \
        name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])

    return out

def resnet_ssd(units, num_stage, filter_list, num_classes, data_type, 
               sizes, ratios, normalizations, steps, num_channels,
               bn_mom=0.9, workspace=512, memonger=False,
               nms_thresh=0.5, force_suppress=False, nms_topk=400):
    
    body = resnet(units, num_stage, filter_list, num_classes, data_type, 
                  bn_mom, workspace, memonger, num_channels)    
    
    graph = mx.viz.plot_network(body.get_internals(), node_attrs={"shape":'rect',"fixedsize":'false'})
    graph.render('img/misc/resnet-plain', cleanup=True)
    
    resnet_arguments = body.get_internals().list_outputs()
    resnet_internals = body.get_internals()

    layer_regex = 'unit1_relu1_output'
    layer_uuids = [layer_uuid for layer_uuid in resnet_arguments 
                                if layer_regex in layer_uuid] + \
                  ['xrelu1_output']
    from_layers = [resnet_internals[layer_uuid] for layer_uuid in layer_uuids]

    body = ssd(from_layers, num_classes, sizes, ratios, normalizations, steps, num_channels, 
               nms_thresh, force_suppress, nms_topk)

    graph = mx.viz.plot_network(body, node_attrs={"shape":'rect',"fixedsize":'false'})
    graph.render('img/misc/resnet-ssd', cleanup=True)
    
    return body