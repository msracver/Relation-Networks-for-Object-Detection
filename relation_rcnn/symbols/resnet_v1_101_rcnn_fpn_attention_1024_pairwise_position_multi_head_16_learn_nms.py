# --------------------------------------------------------
# Relation Networks for Object Detection
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiayuan Gu, Dazhi Cheng
# --------------------------------------------------------

import cPickle
import math
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.box_annotator_ohem import *
from operator_py.nms_multi_target import *
from operator_py.learn_nms import *
from resnet_v1_101_rcnn_learn_nms_base import resnet_v1_101_rcnn_learn_nms_base as NMS_UTILS
# from operator_py.monitor_op import monitor_wrapper


class resnet_v1_101_rcnn_fpn_attention_1024_pairwise_position_multi_head_16_learn_nms(Symbol):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    def get_resnet_v1_fpn_conv(self, data):
        eps = 1e-5
        conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2),
                                      no_bias=True)
        bn_conv1 = mx.symbol.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False, eps=eps)
        scale_conv1 = bn_conv1
        conv1_relu = mx.symbol.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
        pool1 = mx.symbol.Pooling(name='pool1', data=conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                                  stride=(2, 2), pool_type='max')
        res2a_branch1 = mx.symbol.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0),
                                              kernel=(1, 1),
                                              stride=(1, 1), no_bias=True)
        bn2a_branch1 = mx.symbol.BatchNorm(name='bn2a_branch1', data=res2a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=eps)
        scale2a_branch1 = bn2a_branch1
        res2a_branch2a = mx.symbol.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1),
                                               stride=(1, 1), no_bias=True)
        bn2a_branch2a = mx.symbol.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale2a_branch2a = bn2a_branch2a
        res2a_branch2a_relu = mx.symbol.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')
        res2a_branch2b = mx.symbol.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2a_branch2b = mx.symbol.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale2a_branch2b = bn2a_branch2b
        res2a_branch2b_relu = mx.symbol.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
        res2a_branch2c = mx.symbol.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2a_branch2c = mx.symbol.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale2a_branch2c = bn2a_branch2c
        res2a = mx.symbol.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
        res2a_relu = mx.symbol.Activation(name='res2a_relu', data=res2a, act_type='relu')
        res2b_branch2a = mx.symbol.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2a = mx.symbol.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale2b_branch2a = bn2b_branch2a
        res2b_branch2a_relu = mx.symbol.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
        res2b_branch2b = mx.symbol.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2b_branch2b = mx.symbol.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale2b_branch2b = bn2b_branch2b
        res2b_branch2b_relu = mx.symbol.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
        res2b_branch2c = mx.symbol.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2b_branch2c = mx.symbol.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale2b_branch2c = bn2b_branch2c
        res2b = mx.symbol.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
        res2b_relu = mx.symbol.Activation(name='res2b_relu', data=res2b, act_type='relu')
        res2c_branch2a = mx.symbol.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2a = mx.symbol.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale2c_branch2a = bn2c_branch2a
        res2c_branch2a_relu = mx.symbol.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
        res2c_branch2b = mx.symbol.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn2c_branch2b = mx.symbol.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale2c_branch2b = bn2c_branch2b
        res2c_branch2b_relu = mx.symbol.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
        res2c_branch2c = mx.symbol.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn2c_branch2c = mx.symbol.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale2c_branch2c = bn2c_branch2c
        res2c = mx.symbol.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
        res2c_relu = mx.symbol.Activation(name='res2c_relu', data=res2c, act_type='relu')
        res3a_branch1 = mx.symbol.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch1 = mx.symbol.BatchNorm(name='bn3a_branch1', data=res3a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=eps)
        scale3a_branch1 = bn3a_branch1
        res3a_branch2a = mx.symbol.Convolution(name='res3a_branch2a', data=res2c_relu, num_filter=128, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn3a_branch2a = mx.symbol.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale3a_branch2a = bn3a_branch2a
        res3a_branch2a_relu = mx.symbol.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
        res3a_branch2b = mx.symbol.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3a_branch2b = mx.symbol.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale3a_branch2b = bn3a_branch2b
        res3a_branch2b_relu = mx.symbol.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
        res3a_branch2c = mx.symbol.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3a_branch2c = mx.symbol.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale3a_branch2c = bn3a_branch2c
        res3a = mx.symbol.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
        res3a_relu = mx.symbol.Activation(name='res3a_relu', data=res3a, act_type='relu')
        res3b1_branch2a = mx.symbol.Convolution(name='res3b1_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2a = mx.symbol.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale3b1_branch2a = bn3b1_branch2a
        res3b1_branch2a_relu = mx.symbol.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a,
                                                    act_type='relu')
        res3b1_branch2b = mx.symbol.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b1_branch2b = mx.symbol.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale3b1_branch2b = bn3b1_branch2b
        res3b1_branch2b_relu = mx.symbol.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b,
                                                    act_type='relu')
        res3b1_branch2c = mx.symbol.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b1_branch2c = mx.symbol.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale3b1_branch2c = bn3b1_branch2c
        res3b1 = mx.symbol.broadcast_add(name='res3b1', *[res3a_relu, scale3b1_branch2c])
        res3b1_relu = mx.symbol.Activation(name='res3b1_relu', data=res3b1, act_type='relu')
        res3b2_branch2a = mx.symbol.Convolution(name='res3b2_branch2a', data=res3b1_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2a = mx.symbol.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale3b2_branch2a = bn3b2_branch2a
        res3b2_branch2a_relu = mx.symbol.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a,
                                                    act_type='relu')
        res3b2_branch2b = mx.symbol.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu, num_filter=128,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b2_branch2b = mx.symbol.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale3b2_branch2b = bn3b2_branch2b
        res3b2_branch2b_relu = mx.symbol.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b,
                                                    act_type='relu')
        res3b2_branch2c = mx.symbol.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b2_branch2c = mx.symbol.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale3b2_branch2c = bn3b2_branch2c
        res3b2 = mx.symbol.broadcast_add(name='res3b2', *[res3b1_relu, scale3b2_branch2c])
        res3b2_relu = mx.symbol.Activation(name='res3b2_relu', data=res3b2, act_type='relu')
        res3b3_branch2a = mx.symbol.Convolution(name='res3b3_branch2a', data=res3b2_relu, num_filter=128, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2a = mx.symbol.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale3b3_branch2a = bn3b3_branch2a
        res3b3_branch2a_relu = mx.symbol.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a,
                                                    act_type='relu')

        #res3b3_branch2b_offset = mx.symbol.Convolution(name='res3b3_branch2b_offset', data=res3b3_branch2a_relu,
        #                                               num_filter=72, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
        #res3b3_branch2b = mx.contrib.symbol.DeformableConvolution(name='res3b3_branch2b', data=res3b3_branch2a_relu,
        #                                                          offset=res3b3_branch2b_offset,
        #                                                          num_filter=128, pad=(1, 1), kernel=(3, 3),
        #                                                          num_deformable_group=4,
        #                                                          stride=(1, 1), no_bias=True)

        res3b3_branch2b = mx.symbol.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu, num_filter=128,
                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn3b3_branch2b = mx.symbol.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale3b3_branch2b = bn3b3_branch2b
        res3b3_branch2b_relu = mx.symbol.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b,
                                                    act_type='relu')
        res3b3_branch2c = mx.symbol.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu, num_filter=512,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn3b3_branch2c = mx.symbol.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale3b3_branch2c = bn3b3_branch2c
        res3b3 = mx.symbol.broadcast_add(name='res3b3', *[res3b2_relu, scale3b3_branch2c])
        res3b3_relu = mx.symbol.Activation(name='res3b3_relu', data=res3b3, act_type='relu')
        res4a_branch1 = mx.symbol.Convolution(name='res4a_branch1', data=res3b3_relu, num_filter=1024, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch1 = mx.symbol.BatchNorm(name='bn4a_branch1', data=res4a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=eps)
        scale4a_branch1 = bn4a_branch1
        res4a_branch2a = mx.symbol.Convolution(name='res4a_branch2a', data=res3b3_relu, num_filter=256, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn4a_branch2a = mx.symbol.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale4a_branch2a = bn4a_branch2a
        res4a_branch2a_relu = mx.symbol.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
        res4a_branch2b = mx.symbol.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256,
                                               pad=(1, 1),
                                               kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4a_branch2b = mx.symbol.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale4a_branch2b = bn4a_branch2b
        res4a_branch2b_relu = mx.symbol.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
        res4a_branch2c = mx.symbol.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4a_branch2c = mx.symbol.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale4a_branch2c = bn4a_branch2c
        res4a = mx.symbol.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
        res4a_relu = mx.symbol.Activation(name='res4a_relu', data=res4a, act_type='relu')
        res4b1_branch2a = mx.symbol.Convolution(name='res4b1_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2a = mx.symbol.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b1_branch2a = bn4b1_branch2a
        res4b1_branch2a_relu = mx.symbol.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a,
                                                    act_type='relu')
        res4b1_branch2b = mx.symbol.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b1_branch2b = mx.symbol.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b1_branch2b = bn4b1_branch2b
        res4b1_branch2b_relu = mx.symbol.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b,
                                                    act_type='relu')
        res4b1_branch2c = mx.symbol.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b1_branch2c = mx.symbol.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b1_branch2c = bn4b1_branch2c
        res4b1 = mx.symbol.broadcast_add(name='res4b1', *[res4a_relu, scale4b1_branch2c])
        res4b1_relu = mx.symbol.Activation(name='res4b1_relu', data=res4b1, act_type='relu')
        res4b2_branch2a = mx.symbol.Convolution(name='res4b2_branch2a', data=res4b1_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2a = mx.symbol.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b2_branch2a = bn4b2_branch2a
        res4b2_branch2a_relu = mx.symbol.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a,
                                                    act_type='relu')
        res4b2_branch2b = mx.symbol.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b2_branch2b = mx.symbol.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b2_branch2b = bn4b2_branch2b
        res4b2_branch2b_relu = mx.symbol.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b,
                                                    act_type='relu')
        res4b2_branch2c = mx.symbol.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b2_branch2c = mx.symbol.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b2_branch2c = bn4b2_branch2c
        res4b2 = mx.symbol.broadcast_add(name='res4b2', *[res4b1_relu, scale4b2_branch2c])
        res4b2_relu = mx.symbol.Activation(name='res4b2_relu', data=res4b2, act_type='relu')
        res4b3_branch2a = mx.symbol.Convolution(name='res4b3_branch2a', data=res4b2_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2a = mx.symbol.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b3_branch2a = bn4b3_branch2a
        res4b3_branch2a_relu = mx.symbol.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a,
                                                    act_type='relu')
        res4b3_branch2b = mx.symbol.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b3_branch2b = mx.symbol.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b3_branch2b = bn4b3_branch2b
        res4b3_branch2b_relu = mx.symbol.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b,
                                                    act_type='relu')
        res4b3_branch2c = mx.symbol.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b3_branch2c = mx.symbol.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b3_branch2c = bn4b3_branch2c
        res4b3 = mx.symbol.broadcast_add(name='res4b3', *[res4b2_relu, scale4b3_branch2c])
        res4b3_relu = mx.symbol.Activation(name='res4b3_relu', data=res4b3, act_type='relu')
        res4b4_branch2a = mx.symbol.Convolution(name='res4b4_branch2a', data=res4b3_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2a = mx.symbol.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b4_branch2a = bn4b4_branch2a
        res4b4_branch2a_relu = mx.symbol.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a,
                                                    act_type='relu')
        res4b4_branch2b = mx.symbol.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b4_branch2b = mx.symbol.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b4_branch2b = bn4b4_branch2b
        res4b4_branch2b_relu = mx.symbol.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b,
                                                    act_type='relu')
        res4b4_branch2c = mx.symbol.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b4_branch2c = mx.symbol.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b4_branch2c = bn4b4_branch2c
        res4b4 = mx.symbol.broadcast_add(name='res4b4', *[res4b3_relu, scale4b4_branch2c])
        res4b4_relu = mx.symbol.Activation(name='res4b4_relu', data=res4b4, act_type='relu')
        res4b5_branch2a = mx.symbol.Convolution(name='res4b5_branch2a', data=res4b4_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2a = mx.symbol.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b5_branch2a = bn4b5_branch2a
        res4b5_branch2a_relu = mx.symbol.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a,
                                                    act_type='relu')
        res4b5_branch2b = mx.symbol.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b5_branch2b = mx.symbol.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b5_branch2b = bn4b5_branch2b
        res4b5_branch2b_relu = mx.symbol.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b,
                                                    act_type='relu')
        res4b5_branch2c = mx.symbol.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b5_branch2c = mx.symbol.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b5_branch2c = bn4b5_branch2c
        res4b5 = mx.symbol.broadcast_add(name='res4b5', *[res4b4_relu, scale4b5_branch2c])
        res4b5_relu = mx.symbol.Activation(name='res4b5_relu', data=res4b5, act_type='relu')
        res4b6_branch2a = mx.symbol.Convolution(name='res4b6_branch2a', data=res4b5_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2a = mx.symbol.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b6_branch2a = bn4b6_branch2a
        res4b6_branch2a_relu = mx.symbol.Activation(name='res4b6_branch2a_relu', data=scale4b6_branch2a,
                                                    act_type='relu')
        res4b6_branch2b = mx.symbol.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b6_branch2b = mx.symbol.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b6_branch2b = bn4b6_branch2b
        res4b6_branch2b_relu = mx.symbol.Activation(name='res4b6_branch2b_relu', data=scale4b6_branch2b,
                                                    act_type='relu')
        res4b6_branch2c = mx.symbol.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b6_branch2c = mx.symbol.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b6_branch2c = bn4b6_branch2c
        res4b6 = mx.symbol.broadcast_add(name='res4b6', *[res4b5_relu, scale4b6_branch2c])
        res4b6_relu = mx.symbol.Activation(name='res4b6_relu', data=res4b6, act_type='relu')
        res4b7_branch2a = mx.symbol.Convolution(name='res4b7_branch2a', data=res4b6_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2a = mx.symbol.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b7_branch2a = bn4b7_branch2a
        res4b7_branch2a_relu = mx.symbol.Activation(name='res4b7_branch2a_relu', data=scale4b7_branch2a,
                                                    act_type='relu')
        res4b7_branch2b = mx.symbol.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b7_branch2b = mx.symbol.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b7_branch2b = bn4b7_branch2b
        res4b7_branch2b_relu = mx.symbol.Activation(name='res4b7_branch2b_relu', data=scale4b7_branch2b,
                                                    act_type='relu')
        res4b7_branch2c = mx.symbol.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b7_branch2c = mx.symbol.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b7_branch2c = bn4b7_branch2c
        res4b7 = mx.symbol.broadcast_add(name='res4b7', *[res4b6_relu, scale4b7_branch2c])
        res4b7_relu = mx.symbol.Activation(name='res4b7_relu', data=res4b7, act_type='relu')
        res4b8_branch2a = mx.symbol.Convolution(name='res4b8_branch2a', data=res4b7_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2a = mx.symbol.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b8_branch2a = bn4b8_branch2a
        res4b8_branch2a_relu = mx.symbol.Activation(name='res4b8_branch2a_relu', data=scale4b8_branch2a,
                                                    act_type='relu')
        res4b8_branch2b = mx.symbol.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b8_branch2b = mx.symbol.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b8_branch2b = bn4b8_branch2b
        res4b8_branch2b_relu = mx.symbol.Activation(name='res4b8_branch2b_relu', data=scale4b8_branch2b,
                                                    act_type='relu')
        res4b8_branch2c = mx.symbol.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b8_branch2c = mx.symbol.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b8_branch2c = bn4b8_branch2c
        res4b8 = mx.symbol.broadcast_add(name='res4b8', *[res4b7_relu, scale4b8_branch2c])
        res4b8_relu = mx.symbol.Activation(name='res4b8_relu', data=res4b8, act_type='relu')
        res4b9_branch2a = mx.symbol.Convolution(name='res4b9_branch2a', data=res4b8_relu, num_filter=256, pad=(0, 0),
                                                kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2a = mx.symbol.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b9_branch2a = bn4b9_branch2a
        res4b9_branch2a_relu = mx.symbol.Activation(name='res4b9_branch2a_relu', data=scale4b9_branch2a,
                                                    act_type='relu')
        res4b9_branch2b = mx.symbol.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu, num_filter=256,
                                                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b9_branch2b = mx.symbol.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b9_branch2b = bn4b9_branch2b
        res4b9_branch2b_relu = mx.symbol.Activation(name='res4b9_branch2b_relu', data=scale4b9_branch2b,
                                                    act_type='relu')
        res4b9_branch2c = mx.symbol.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu, num_filter=1024,
                                                pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b9_branch2c = mx.symbol.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c, use_global_stats=True,
                                             fix_gamma=False, eps=eps)
        scale4b9_branch2c = bn4b9_branch2c
        res4b9 = mx.symbol.broadcast_add(name='res4b9', *[res4b8_relu, scale4b9_branch2c])
        res4b9_relu = mx.symbol.Activation(name='res4b9_relu', data=res4b9, act_type='relu')
        res4b10_branch2a = mx.symbol.Convolution(name='res4b10_branch2a', data=res4b9_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2a = mx.symbol.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b10_branch2a = bn4b10_branch2a
        res4b10_branch2a_relu = mx.symbol.Activation(name='res4b10_branch2a_relu', data=scale4b10_branch2a,
                                                     act_type='relu')
        res4b10_branch2b = mx.symbol.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b10_branch2b = mx.symbol.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b10_branch2b = bn4b10_branch2b
        res4b10_branch2b_relu = mx.symbol.Activation(name='res4b10_branch2b_relu', data=scale4b10_branch2b,
                                                     act_type='relu')
        res4b10_branch2c = mx.symbol.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b10_branch2c = mx.symbol.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b10_branch2c = bn4b10_branch2c
        res4b10 = mx.symbol.broadcast_add(name='res4b10', *[res4b9_relu, scale4b10_branch2c])
        res4b10_relu = mx.symbol.Activation(name='res4b10_relu', data=res4b10, act_type='relu')
        res4b11_branch2a = mx.symbol.Convolution(name='res4b11_branch2a', data=res4b10_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2a = mx.symbol.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b11_branch2a = bn4b11_branch2a
        res4b11_branch2a_relu = mx.symbol.Activation(name='res4b11_branch2a_relu', data=scale4b11_branch2a,
                                                     act_type='relu')
        res4b11_branch2b = mx.symbol.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b11_branch2b = mx.symbol.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b11_branch2b = bn4b11_branch2b
        res4b11_branch2b_relu = mx.symbol.Activation(name='res4b11_branch2b_relu', data=scale4b11_branch2b,
                                                     act_type='relu')
        res4b11_branch2c = mx.symbol.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b11_branch2c = mx.symbol.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b11_branch2c = bn4b11_branch2c
        res4b11 = mx.symbol.broadcast_add(name='res4b11', *[res4b10_relu, scale4b11_branch2c])
        res4b11_relu = mx.symbol.Activation(name='res4b11_relu', data=res4b11, act_type='relu')
        res4b12_branch2a = mx.symbol.Convolution(name='res4b12_branch2a', data=res4b11_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2a = mx.symbol.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b12_branch2a = bn4b12_branch2a
        res4b12_branch2a_relu = mx.symbol.Activation(name='res4b12_branch2a_relu', data=scale4b12_branch2a,
                                                     act_type='relu')
        res4b12_branch2b = mx.symbol.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b12_branch2b = mx.symbol.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b12_branch2b = bn4b12_branch2b
        res4b12_branch2b_relu = mx.symbol.Activation(name='res4b12_branch2b_relu', data=scale4b12_branch2b,
                                                     act_type='relu')
        res4b12_branch2c = mx.symbol.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b12_branch2c = mx.symbol.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b12_branch2c = bn4b12_branch2c
        res4b12 = mx.symbol.broadcast_add(name='res4b12', *[res4b11_relu, scale4b12_branch2c])
        res4b12_relu = mx.symbol.Activation(name='res4b12_relu', data=res4b12, act_type='relu')
        res4b13_branch2a = mx.symbol.Convolution(name='res4b13_branch2a', data=res4b12_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2a = mx.symbol.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b13_branch2a = bn4b13_branch2a
        res4b13_branch2a_relu = mx.symbol.Activation(name='res4b13_branch2a_relu', data=scale4b13_branch2a,
                                                     act_type='relu')
        res4b13_branch2b = mx.symbol.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b13_branch2b = mx.symbol.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b13_branch2b = bn4b13_branch2b
        res4b13_branch2b_relu = mx.symbol.Activation(name='res4b13_branch2b_relu', data=scale4b13_branch2b,
                                                     act_type='relu')
        res4b13_branch2c = mx.symbol.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b13_branch2c = mx.symbol.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b13_branch2c = bn4b13_branch2c
        res4b13 = mx.symbol.broadcast_add(name='res4b13', *[res4b12_relu, scale4b13_branch2c])
        res4b13_relu = mx.symbol.Activation(name='res4b13_relu', data=res4b13, act_type='relu')
        res4b14_branch2a = mx.symbol.Convolution(name='res4b14_branch2a', data=res4b13_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2a = mx.symbol.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b14_branch2a = bn4b14_branch2a
        res4b14_branch2a_relu = mx.symbol.Activation(name='res4b14_branch2a_relu', data=scale4b14_branch2a,
                                                     act_type='relu')
        res4b14_branch2b = mx.symbol.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b14_branch2b = mx.symbol.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b14_branch2b = bn4b14_branch2b
        res4b14_branch2b_relu = mx.symbol.Activation(name='res4b14_branch2b_relu', data=scale4b14_branch2b,
                                                     act_type='relu')
        res4b14_branch2c = mx.symbol.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b14_branch2c = mx.symbol.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b14_branch2c = bn4b14_branch2c
        res4b14 = mx.symbol.broadcast_add(name='res4b14', *[res4b13_relu, scale4b14_branch2c])
        res4b14_relu = mx.symbol.Activation(name='res4b14_relu', data=res4b14, act_type='relu')
        res4b15_branch2a = mx.symbol.Convolution(name='res4b15_branch2a', data=res4b14_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2a = mx.symbol.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b15_branch2a = bn4b15_branch2a
        res4b15_branch2a_relu = mx.symbol.Activation(name='res4b15_branch2a_relu', data=scale4b15_branch2a,
                                                     act_type='relu')
        res4b15_branch2b = mx.symbol.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b15_branch2b = mx.symbol.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b15_branch2b = bn4b15_branch2b
        res4b15_branch2b_relu = mx.symbol.Activation(name='res4b15_branch2b_relu', data=scale4b15_branch2b,
                                                     act_type='relu')
        res4b15_branch2c = mx.symbol.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b15_branch2c = mx.symbol.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b15_branch2c = bn4b15_branch2c
        res4b15 = mx.symbol.broadcast_add(name='res4b15', *[res4b14_relu, scale4b15_branch2c])
        res4b15_relu = mx.symbol.Activation(name='res4b15_relu', data=res4b15, act_type='relu')
        res4b16_branch2a = mx.symbol.Convolution(name='res4b16_branch2a', data=res4b15_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2a = mx.symbol.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b16_branch2a = bn4b16_branch2a
        res4b16_branch2a_relu = mx.symbol.Activation(name='res4b16_branch2a_relu', data=scale4b16_branch2a,
                                                     act_type='relu')
        res4b16_branch2b = mx.symbol.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b16_branch2b = mx.symbol.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b16_branch2b = bn4b16_branch2b
        res4b16_branch2b_relu = mx.symbol.Activation(name='res4b16_branch2b_relu', data=scale4b16_branch2b,
                                                     act_type='relu')
        res4b16_branch2c = mx.symbol.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b16_branch2c = mx.symbol.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b16_branch2c = bn4b16_branch2c
        res4b16 = mx.symbol.broadcast_add(name='res4b16', *[res4b15_relu, scale4b16_branch2c])
        res4b16_relu = mx.symbol.Activation(name='res4b16_relu', data=res4b16, act_type='relu')
        res4b17_branch2a = mx.symbol.Convolution(name='res4b17_branch2a', data=res4b16_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2a = mx.symbol.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b17_branch2a = bn4b17_branch2a
        res4b17_branch2a_relu = mx.symbol.Activation(name='res4b17_branch2a_relu', data=scale4b17_branch2a,
                                                     act_type='relu')
        res4b17_branch2b = mx.symbol.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b17_branch2b = mx.symbol.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b17_branch2b = bn4b17_branch2b
        res4b17_branch2b_relu = mx.symbol.Activation(name='res4b17_branch2b_relu', data=scale4b17_branch2b,
                                                     act_type='relu')
        res4b17_branch2c = mx.symbol.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b17_branch2c = mx.symbol.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b17_branch2c = bn4b17_branch2c
        res4b17 = mx.symbol.broadcast_add(name='res4b17', *[res4b16_relu, scale4b17_branch2c])
        res4b17_relu = mx.symbol.Activation(name='res4b17_relu', data=res4b17, act_type='relu')
        res4b18_branch2a = mx.symbol.Convolution(name='res4b18_branch2a', data=res4b17_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2a = mx.symbol.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b18_branch2a = bn4b18_branch2a
        res4b18_branch2a_relu = mx.symbol.Activation(name='res4b18_branch2a_relu', data=scale4b18_branch2a,
                                                     act_type='relu')
        res4b18_branch2b = mx.symbol.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b18_branch2b = mx.symbol.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b18_branch2b = bn4b18_branch2b
        res4b18_branch2b_relu = mx.symbol.Activation(name='res4b18_branch2b_relu', data=scale4b18_branch2b,
                                                     act_type='relu')
        res4b18_branch2c = mx.symbol.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b18_branch2c = mx.symbol.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b18_branch2c = bn4b18_branch2c
        res4b18 = mx.symbol.broadcast_add(name='res4b18', *[res4b17_relu, scale4b18_branch2c])
        res4b18_relu = mx.symbol.Activation(name='res4b18_relu', data=res4b18, act_type='relu')
        res4b19_branch2a = mx.symbol.Convolution(name='res4b19_branch2a', data=res4b18_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2a = mx.symbol.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b19_branch2a = bn4b19_branch2a
        res4b19_branch2a_relu = mx.symbol.Activation(name='res4b19_branch2a_relu', data=scale4b19_branch2a,
                                                     act_type='relu')
        res4b19_branch2b = mx.symbol.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b19_branch2b = mx.symbol.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b19_branch2b = bn4b19_branch2b
        res4b19_branch2b_relu = mx.symbol.Activation(name='res4b19_branch2b_relu', data=scale4b19_branch2b,
                                                     act_type='relu')
        res4b19_branch2c = mx.symbol.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b19_branch2c = mx.symbol.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b19_branch2c = bn4b19_branch2c
        res4b19 = mx.symbol.broadcast_add(name='res4b19', *[res4b18_relu, scale4b19_branch2c])
        res4b19_relu = mx.symbol.Activation(name='res4b19_relu', data=res4b19, act_type='relu')
        res4b20_branch2a = mx.symbol.Convolution(name='res4b20_branch2a', data=res4b19_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2a = mx.symbol.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b20_branch2a = bn4b20_branch2a
        res4b20_branch2a_relu = mx.symbol.Activation(name='res4b20_branch2a_relu', data=scale4b20_branch2a,
                                                     act_type='relu')
        res4b20_branch2b = mx.symbol.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b20_branch2b = mx.symbol.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b20_branch2b = bn4b20_branch2b
        res4b20_branch2b_relu = mx.symbol.Activation(name='res4b20_branch2b_relu', data=scale4b20_branch2b,
                                                     act_type='relu')
        res4b20_branch2c = mx.symbol.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b20_branch2c = mx.symbol.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b20_branch2c = bn4b20_branch2c
        res4b20 = mx.symbol.broadcast_add(name='res4b20', *[res4b19_relu, scale4b20_branch2c])
        res4b20_relu = mx.symbol.Activation(name='res4b20_relu', data=res4b20, act_type='relu')
        res4b21_branch2a = mx.symbol.Convolution(name='res4b21_branch2a', data=res4b20_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2a = mx.symbol.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b21_branch2a = bn4b21_branch2a
        res4b21_branch2a_relu = mx.symbol.Activation(name='res4b21_branch2a_relu', data=scale4b21_branch2a,
                                                     act_type='relu')
        res4b21_branch2b = mx.symbol.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu, num_filter=256,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
        bn4b21_branch2b = mx.symbol.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b21_branch2b = bn4b21_branch2b
        res4b21_branch2b_relu = mx.symbol.Activation(name='res4b21_branch2b_relu', data=scale4b21_branch2b,
                                                     act_type='relu')
        res4b21_branch2c = mx.symbol.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b21_branch2c = mx.symbol.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b21_branch2c = bn4b21_branch2c
        res4b21 = mx.symbol.broadcast_add(name='res4b21', *[res4b20_relu, scale4b21_branch2c])
        res4b21_relu = mx.symbol.Activation(name='res4b21_relu', data=res4b21, act_type='relu')
        res4b22_branch2a = mx.symbol.Convolution(name='res4b22_branch2a', data=res4b21_relu, num_filter=256, pad=(0, 0),
                                                 kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2a = mx.symbol.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b22_branch2a = bn4b22_branch2a
        res4b22_branch2a_relu = mx.symbol.Activation(name='res4b22_branch2a_relu', data=scale4b22_branch2a,
                                                     act_type='relu')

        res4b22_branch2b = mx.symbol.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu, num_filter=256,
                pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)

        # res4b22_branch2b_offset = mx.symbol.Convolution(name='res4b22_branch2b_offset', data=res4b22_branch2a_relu,
        #                                                 num_filter=72, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
        # res4b22_branch2b = mx.contrib.symbol.DeformableConvolution(name='res4b22_branch2b', data=res4b22_branch2a_relu,
        #                                                            offset=res4b22_branch2b_offset,
        #                                                            num_filter=256, pad=(1, 1), kernel=(3, 3),
        #                                                            num_deformable_group=4,
        #                                                            stride=(1, 1), no_bias=True)

        bn4b22_branch2b = mx.symbol.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b22_branch2b = bn4b22_branch2b
        res4b22_branch2b_relu = mx.symbol.Activation(name='res4b22_branch2b_relu', data=scale4b22_branch2b,
                                                     act_type='relu')
        res4b22_branch2c = mx.symbol.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu, num_filter=1024,
                                                 pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn4b22_branch2c = mx.symbol.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c, use_global_stats=True,
                                              fix_gamma=False, eps=eps)
        scale4b22_branch2c = bn4b22_branch2c
        res4b22 = mx.symbol.broadcast_add(name='res4b22', *[res4b21_relu, scale4b22_branch2c])
        res4b22_relu = mx.symbol.Activation(name='res4b22_relu', data=res4b22, act_type='relu')

        # ------------------------------ END OF CONV4 ------------------------
        res5a_branch1 = mx.symbol.Convolution(name='res5a_branch1', data=res4b22_relu, num_filter=2048, pad=(0, 0),
                                              kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn5a_branch1 = mx.symbol.BatchNorm(name='bn5a_branch1', data=res5a_branch1, use_global_stats=True,
                                           fix_gamma=False, eps=eps)
        scale5a_branch1 = bn5a_branch1
        res5a_branch2a = mx.symbol.Convolution(name='res5a_branch2a', data=res4b22_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(2, 2), no_bias=True)
        bn5a_branch2a = mx.symbol.BatchNorm(name='bn5a_branch2a', data=res5a_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale5a_branch2a = bn5a_branch2a
        res5a_branch2a_relu = mx.symbol.Activation(name='res5a_branch2a_relu', data=scale5a_branch2a, act_type='relu')
        res5a_branch2b = mx.symbol.Convolution(name='res5a_branch2b', data=res5a_branch2a_relu, num_filter=512, pad=(1, 1),
                kernel=(3, 3), stride=(1, 1), no_bias=True, cudnn_off=True)

        # deconv 5a
        # res5a_branch2b_offset = mx.symbol.Convolution(name='res5a_branch2b_offset', data=res5a_branch2a_relu,
        #                                               num_filter=72, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
        # res5a_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5a_branch2b', data=res5a_branch2a_relu,
        #                                                          offset=res5a_branch2b_offset,
        #                                                          num_filter=512, pad=(1, 1), kernel=(3, 3),
        #                                                          num_deformable_group=4,
        #                                                          stride=(1, 1), no_bias=True)

        bn5a_branch2b = mx.symbol.BatchNorm(name='bn5a_branch2b', data=res5a_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale5a_branch2b = bn5a_branch2b
        res5a_branch2b_relu = mx.symbol.Activation(name='res5a_branch2b_relu', data=scale5a_branch2b, act_type='relu')
        res5a_branch2c = mx.symbol.Convolution(name='res5a_branch2c', data=res5a_branch2b_relu, num_filter=2048,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5a_branch2c = mx.symbol.BatchNorm(name='bn5a_branch2c', data=res5a_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale5a_branch2c = bn5a_branch2c
        res5a = mx.symbol.broadcast_add(name='res5a', *[scale5a_branch1, scale5a_branch2c])
        res5a_relu = mx.symbol.Activation(name='res5a_relu', data=res5a, act_type='relu')
        res5b_branch2a = mx.symbol.Convolution(name='res5b_branch2a', data=res5a_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2a = mx.symbol.BatchNorm(name='bn5b_branch2a', data=res5b_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale5b_branch2a = bn5b_branch2a
        res5b_branch2a_relu = mx.symbol.Activation(name='res5b_branch2a_relu', data=scale5b_branch2a, act_type='relu')
        res5b_branch2b = mx.symbol.Convolution(name='res5b_branch2b', data=res5b_branch2a_relu, num_filter=512, pad=(1, 1),
               kernel=(3, 3), stride=(1, 1), no_bias=True, cudnn_off=True)

        # dconv 5b
        # res5b_branch2b_offset = mx.symbol.Convolution(name='res5b_branch2b_offset', data=res5b_branch2a_relu,
        #                                               num_filter=72, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
        # res5b_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5b_branch2b', data=res5b_branch2a_relu,
        #                                                          offset=res5b_branch2b_offset,
        #                                                          num_filter=512, pad=(1, 1), kernel=(3, 3),
        #                                                          num_deformable_group=4,
        #                                                          stride=(1, 1), no_bias=True)

        bn5b_branch2b = mx.symbol.BatchNorm(name='bn5b_branch2b', data=res5b_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale5b_branch2b = bn5b_branch2b
        res5b_branch2b_relu = mx.symbol.Activation(name='res5b_branch2b_relu', data=scale5b_branch2b, act_type='relu')
        res5b_branch2c = mx.symbol.Convolution(name='res5b_branch2c', data=res5b_branch2b_relu, num_filter=2048,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5b_branch2c = mx.symbol.BatchNorm(name='bn5b_branch2c', data=res5b_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale5b_branch2c = bn5b_branch2c
        res5b = mx.symbol.broadcast_add(name='res5b', *[res5a_relu, scale5b_branch2c])
        res5b_relu = mx.symbol.Activation(name='res5b_relu', data=res5b, act_type='relu')
        res5c_branch2a = mx.symbol.Convolution(name='res5c_branch2a', data=res5b_relu, num_filter=512, pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2a = mx.symbol.BatchNorm(name='bn5c_branch2a', data=res5c_branch2a, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale5c_branch2a = bn5c_branch2a
        res5c_branch2a_relu = mx.symbol.Activation(name='res5c_branch2a_relu', data=scale5c_branch2a, act_type='relu')
        res5c_branch2b = mx.symbol.Convolution(name='res5c_branch2b', data=res5c_branch2a_relu, num_filter=512, pad=(1, 1),
               kernel=(3, 3), stride=(1, 1), no_bias=True, cudnn_off=True)

        # res5c_branch2b_offset = mx.symbol.Convolution(name='res5c_branch2b_offset', data=res5c_branch2a_relu,
        #                                               num_filter=72, pad=(1, 1), kernel=(3, 3), stride=(1, 1))
        # res5c_branch2b = mx.contrib.symbol.DeformableConvolution(name='res5c_branch2b', data=res5c_branch2a_relu,
        #                                                          offset=res5c_branch2b_offset,
        #                                                          num_filter=512, pad=(1, 1), kernel=(3, 3),
        #                                                          num_deformable_group=4,
        #                                                          stride=(1, 1), no_bias=True)

        bn5c_branch2b = mx.symbol.BatchNorm(name='bn5c_branch2b', data=res5c_branch2b, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale5c_branch2b = bn5c_branch2b
        res5c_branch2b_relu = mx.symbol.Activation(name='res5c_branch2b_relu', data=scale5c_branch2b, act_type='relu')

        res5c_branch2c = mx.symbol.Convolution(name='res5c_branch2c', data=res5c_branch2b_relu, num_filter=2048,
                                               pad=(0, 0),
                                               kernel=(1, 1), stride=(1, 1), no_bias=True)
        bn5c_branch2c = mx.symbol.BatchNorm(name='bn5c_branch2c', data=res5c_branch2c, use_global_stats=True,
                                            fix_gamma=False, eps=eps)
        scale5c_branch2c = bn5c_branch2c
        res5c = mx.symbol.broadcast_add(name='res5c', *[res5b_relu, scale5c_branch2c])
        res5c_relu = mx.symbol.Activation(name='res5c_relu', data=res5c, act_type='relu')

        # fpn
        fpn_ft32_1x1 = mx.symbol.Convolution(
            data=res5c_relu, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=256, name="fpn_ft32_1x1")
        fpn_ft16_1x1 = mx.symbol.Convolution(
            data=res4b22_relu, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=256, name="fpn_ft16_1x1")
        fpn_ft8_1x1 = mx.symbol.Convolution(
            data=res3b3_relu, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=256, name="fpn_ft8_1x1")
        fpn_ft4_1x1 = mx.symbol.Convolution(
            data=res2c_relu, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=256, name="fpn_ft4_1x1")

        fpn_ft32_upsample = mx.symbol.UpSampling(
            fpn_ft32_1x1, scale=2, sample_type='nearest', name="fpn_ft32_upsample")
        # fpn_ft32_upsample = mx.symbol.Crop(*[fpn_ft32_upsample, fpn_ft16_1x1], offset=(0, 0), name="fpn_ft32_crop")
        fpn_ft16_plus = mx.sym.ElementWiseSum(*[fpn_ft32_upsample, fpn_ft16_1x1], name='fpn_ft16_plus')

        fpn_ft16_upsample = mx.symbol.UpSampling(
            fpn_ft16_plus, scale=2, sample_type='nearest', name="fpn_ft16_upsample")
        # fpn_ft16_upsample = mx.symbol.Crop(*[fpn_ft16_upsample, fpn_ft8_1x1], offset=(0, 0), name="fpn_ft16_crop")
        fpn_ft8_plus = mx.sym.ElementWiseSum(*[fpn_ft16_upsample, fpn_ft8_1x1], name='fpn_ft8_plus')

        fpn_ft8_upsample = mx.symbol.UpSampling(
            fpn_ft8_plus, scale=2, sample_type='nearest', name="fpn_ft8_upsample")
        # fpn_ft8_upsample = mx.symbol.Crop(*[fpn_ft8_upsample, fpn_ft4_1x1], offset=(0, 0), name="fpn_ft8_crop")
        fpn_ft4_plus = mx.sym.ElementWiseSum(*[fpn_ft8_upsample, fpn_ft4_1x1], name='fpn_ft4_plus')

        fpn_ft64_3x3 = mx.symbol.Convolution(
            data=fpn_ft32_1x1, kernel=(3, 3), pad=(1, 1), stride=(2, 2), num_filter=256, name="fpn_ft64_3x3")
        fpn_ft32_3x3 = mx.symbol.Convolution(
            data=fpn_ft32_1x1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=256, name="fpn_ft32_3x3")
        fpn_ft16_3x3 = mx.symbol.Convolution(
            data=fpn_ft16_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=256, name="fpn_ft16_3x3")
        fpn_ft8_3x3 = mx.symbol.Convolution(
            data=fpn_ft8_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=256, name="fpn_ft8_3x3")
        fpn_ft4_3x3 = mx.symbol.Convolution(
            data=fpn_ft4_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=256, name="fpn_ft4_3x3")

        return fpn_ft64_3x3, fpn_ft32_3x3, fpn_ft16_3x3, fpn_ft8_3x3, fpn_ft4_3x3

    @staticmethod
    def extract_position_embedding(position_mat, feat_dim, wave_length=1000):
        # position_mat, [num_rois, nongt_dim, 4]
        feat_range = mx.sym.arange(0, feat_dim / 8)
        dim_mat = mx.sym.broadcast_power(lhs=mx.sym.full((1,), wave_length),
                                         rhs=(8. / feat_dim) * feat_range)
        dim_mat = mx.sym.Reshape(dim_mat, shape=(1, 1, 1, -1))
        position_mat = mx.sym.expand_dims(100.0 * position_mat, axis=3)
        div_mat = mx.sym.broadcast_div(lhs=position_mat, rhs=dim_mat)
        sin_mat = mx.sym.sin(data=div_mat)
        cos_mat = mx.sym.cos(data=div_mat)
        # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
        embedding = mx.sym.concat(sin_mat, cos_mat, dim=3)
        # embedding, [num_rois, nongt_dim, feat_dim]
        embedding = mx.sym.Reshape(embedding, shape=(0, 0, feat_dim))
        return embedding

    @staticmethod
    def extract_position_matrix(bbox, non_gt_index):
        """ Extract position matrix

        Args:
            bbox: [num_boxes, 4]

        Returns:
            position_matrix: [num_boxes, nongt_dim, 4]
        """
        xmin, ymin, xmax, ymax = mx.sym.split(data=bbox, num_outputs=4, axis=1)
        # [num_boxes, 1]
        bbox_width = xmax - xmin + 1.
        bbox_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        # [nongt_dim, 1]
        if non_gt_index is None:
            bbox_width_nongt = bbox_width
            bbox_height_nongt = bbox_height
            center_x_nongt = center_x
            center_y_nongt = center_y
        else:
            bbox_width_nongt = mx.symbol.take(a=bbox_width, indices=non_gt_index, axis=0)
            bbox_height_nongt = mx.symbol.take(a=bbox_height, indices=non_gt_index, axis=0)
            center_x_nongt = mx.symbol.take(a=center_x, indices=non_gt_index, axis=0)
            center_y_nongt = mx.symbol.take(a=center_y, indices=non_gt_index, axis=0)
        # [num_boxes, nongt_dim]
        delta_x = mx.sym.broadcast_minus(lhs=center_x,
                                         rhs=mx.sym.transpose(center_x_nongt))
        delta_x = mx.sym.broadcast_div(delta_x, bbox_width)
        delta_x = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_x), 1e-3))
        delta_y = mx.sym.broadcast_minus(lhs=center_y,
                                         rhs=mx.sym.transpose(center_y_nongt))
        delta_y = mx.sym.broadcast_div(delta_y, bbox_height)
        delta_y = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_y), 1e-3))
        delta_width = mx.sym.broadcast_div(lhs=bbox_width,
                                           rhs=mx.sym.transpose(bbox_width_nongt))
        delta_width = mx.sym.log(delta_width)
        delta_height = mx.sym.broadcast_div(lhs=bbox_height,
                                            rhs=mx.sym.transpose(bbox_height_nongt))
        delta_height = mx.sym.log(delta_height)
        concat_list = [delta_x, delta_y, delta_width, delta_height]
        for idx, sym in enumerate(concat_list):
            concat_list[idx] = mx.sym.expand_dims(sym, axis=2)
        position_matrix = mx.sym.concat(*concat_list, dim=2)
        return position_matrix

    def attention_module_multi_head(self, roi_feat, position_embedding,
                                    non_gt_index, fc_dim, feat_dim,
                                    dim=(1024, 1024, 1024),
                                    group=16, index=1):
        """ Attetion module with vectorized version

        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [1, emb_dim, num_rois, nongt_dim]
            non_gt_index:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:

        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        # non_gt_index = monitor_wrapper(non_gt_index, 'non_gt_index')
        if non_gt_index is None:
            nongt_roi_feat = roi_feat
        else:
            nongt_roi_feat = mx.symbol.take(a=roi_feat, indices=non_gt_index, axis=0)

        # [1, emb_dim, num_rois, nongt_dim]
        # position_feat_1, [1, fc_dim, num_rois, nongt_dim]
        position_feat_1 = mx.sym.Convolution(name='pair_pos_fc1_' + str(index),
                                             data=position_embedding, num_filter=fc_dim,
                                             kernel=(1, 1), stride=(1, 1), pad=(0, 0))
        position_feat_1_relu = mx.sym.Activation(data=position_feat_1, act_type='relu')
        # aff_weight, [num_rois, fc_dim, nongt_dim, 1]
        aff_weight = mx.sym.transpose(position_feat_1_relu, axes=(2, 1, 3, 0))
        # aff_weight, [num_rois, fc_dim, nongt_dim]
        aff_weight = mx.sym.Reshape(aff_weight, shape=(0, 0, 0))

        # multi head
        assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
        q_data = mx.sym.FullyConnected(name='query_' + str(index),
                                       data=roi_feat,
                                       num_hidden=dim[0])
        q_data_batch = mx.sym.Reshape(q_data, shape=(-1, group, dim_group[0]))
        q_data_batch = mx.sym.transpose(q_data_batch, axes=(1, 0, 2))
        k_data = mx.symbol.FullyConnected(name='key_' + str(index),
                                          data=nongt_roi_feat,
                                          num_hidden=dim[1])
        k_data_batch = mx.sym.Reshape(k_data, shape=(-1, group, dim_group[1]))
        k_data_batch = mx.sym.transpose(k_data_batch, axes=(1, 0, 2))
        v_data = nongt_roi_feat
        # v_data =  mx.symbol.FullyConnected(name='value_'+str(index)+'_'+str(gid), data=roi_feat, num_hidden=dim_group[2])
        aff = mx.symbol.batch_dot(lhs=q_data_batch, rhs=k_data_batch, transpose_a=False, transpose_b=True)
        # aff_scale, [group, num_rois, nongt_dim]
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        aff_scale = mx.sym.transpose(aff_scale, axes=(1, 0, 2))

        assert fc_dim == group, 'fc_dim != group'
        # weighted_aff, [num_rois, fc_dim, nongt_dim]
        weighted_aff = mx.sym.log(mx.sym.maximum(left=aff_weight, right=1e-6)) + aff_scale
        aff_softmax = mx.symbol.softmax(data=weighted_aff, axis=2, name='softmax_' + str(index))
        # [num_rois * fc_dim, nongt_dim]
        aff_softmax_reshape = mx.sym.Reshape(aff_softmax, shape=(-3, -2))
        # output_t, [num_rois * fc_dim, feat_dim]
        output_t = mx.symbol.dot(lhs=aff_softmax_reshape, rhs=v_data)
        # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
        output_t = mx.sym.Reshape(output_t, shape=(-1, fc_dim * feat_dim, 1, 1))
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = mx.symbol.Convolution(name='linear_out_' + str(index), data=output_t,
                                           kernel=(1, 1), num_filter=dim[2], num_group=fc_dim)
        output = mx.sym.Reshape(linear_out, shape=(0, 0))
        return output

    def attention_module_nms_multi_head(self,
                                        roi_feat, position_mat, num_rois,
                                        dim=(1024, 1024, 1024), fc_dim=(64, 16), feat_dim=1024,
                                        group=16, index=1):
        """ Attetion module with vectorized version

        Args:
            roi_feat: [num_rois, num_fg_classes, feat_dim]
            position_mat: [num_fg_classes, num_rois, num_rois, 4]
            num_rois: number of rois
            dim: key, query and linear_out dim
            fc_dim:
            feat_dim:
            group:
            index:

        Returns:
            output: [num_rois, num_fg_classes, fc_dim]
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        roi_feat = mx.sym.transpose(roi_feat, axes=(1, 0, 2))
        # roi_feat_reshape, [num_fg_classes*num_rois, feat_dim]
        roi_feat_reshape = mx.sym.Reshape(roi_feat, shape=(-3, -2))
        # position_embedding, [num_fg_classes, num_rois, num_rois, fc_dim[0]]
        position_embedding = NMS_UTILS.extract_pairwise_multi_position_embedding(position_mat, fc_dim[0])
        # [num_fg_classes * num_rois * num_rois, fc_dim[0]]
        position_embedding_reshape =  mx.sym.Reshape(position_embedding, shape=(-1, fc_dim[0]))
        # position_feat_1, [num_fg_classes * num_rois * num_rois, fc_dim[1]]
        position_feat_1 = mx.sym.FullyConnected(name='nms_pair_pos_fc1_' + str(index),
                                                data=position_embedding_reshape,
                                                num_hidden=fc_dim[1])
        # position_feat_1, [num_fg_classes, num_rois, num_rois, fc_dim[1]]
        position_feat_1 = mx.sym.Reshape(position_feat_1, shape=(-1, num_rois, num_rois, fc_dim[1]))
        aff_weight = mx.sym.Activation(data=position_feat_1, act_type='relu')
        # aff_weight, [num_fg_classes, fc_dim[1], num_rois, num_rois]
        aff_weight = mx.sym.transpose(aff_weight, axes=(0, 3, 1, 2))

        ####################### multi head in batch###########################
        assert dim[0] == dim[1], 'Matrix multi requires the same dims!'
        # q_data, [num_fg_classes * num_rois, dim[0]]
        q_data = mx.sym.FullyConnected(name='nms_query_' + str(index), data=roi_feat_reshape, num_hidden=dim[0])
        # q_data, [num_fg_classes, num_rois, group, dim_group[0]]
        q_data_batch = mx.sym.Reshape(q_data, shape=(-1, num_rois, group, dim_group[0]))
        q_data_batch = mx.sym.transpose(q_data_batch, axes=(0, 2, 1, 3))
        # q_data_batch, [num_fg_classes * group, num_rois, dim_group[0]]
        q_data_batch = mx.sym.Reshape(q_data_batch, shape=(-3, -2))
        k_data = mx.sym.FullyConnected(name='nms_key_' + str(index), data=roi_feat_reshape, num_hidden=dim[1])
        # k_data, [num_fg_classes, num_rois, group, dim_group[1]]
        k_data_batch = mx.sym.Reshape(k_data, shape=(-1, num_rois, group, dim_group[1]))
        k_data_batch = mx.sym.transpose(k_data_batch, axes=(0, 2, 1, 3))
        # k_data_batch, [num_fg_classes * group, num_rois, dim_group[1]]
        k_data_batch = mx.sym.Reshape(k_data_batch, shape=(-3, -2))
        v_data = roi_feat
        aff = mx.symbol.batch_dot(lhs=q_data_batch, rhs=k_data_batch, transpose_a=False, transpose_b=True)
        # aff_scale, [num_fg_classes * group, num_rois, num_rois]
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff

        assert fc_dim[1] == group, 'Check the dimensions in attention!'
        # [num_fg_classes * fc_dim[1], num_rois, num_rois]
        aff_weight_reshape = mx.sym.Reshape(aff_weight, shape=(-3, -2))
        # weighted_aff, [num_fg_classes * fc_dim[1], num_rois, num_rois]
        weighted_aff= mx.sym.log(mx.sym.maximum(left=aff_weight_reshape, right=1e-6)) + aff_scale
        # aff_softmax, [num_fg_classes * fc_dim[1], num_rois, num_rois]
        aff_softmax = mx.symbol.softmax(data=weighted_aff, axis=2, name='nms_softmax_' + str(index))
        aff_softmax_reshape = mx.sym.Reshape(aff_softmax, shape=(-1, fc_dim[1] * num_rois, 0))
        # output_t, [num_fg_classes, fc_dim[1] * num_rois, feat_dim]
        output_t = mx.symbol.batch_dot(lhs=aff_softmax_reshape, rhs=v_data)
        # output_t_reshape, [num_fg_classes, fc_dim[1], num_rois, feat_dim]
        output_t_reshape = mx.sym.Reshape(output_t, shape=(-1, fc_dim[1], num_rois, feat_dim))
        # output_t_reshape, [fc_dim[1], feat_dim, num_rois, num_fg_classes]
        output_t_reshape = mx.sym.transpose(output_t_reshape, axes=(1, 3, 2, 0))
        # output_t_reshape, [1, fc_dim[1] * feat_dim, num_rois, num_fg_classes]
        output_t_reshape = mx.sym.Reshape(output_t_reshape, shape=(1, fc_dim[1] * feat_dim, num_rois, -1))
        linear_out = mx.symbol.Convolution(name='nms_linear_out_' + str(index),
                                           data=output_t_reshape,
                                           kernel=(1, 1), num_filter=dim[2], num_group=fc_dim[1])
        # [dim[2], num_rois, num_fg_classes]
        linear_out_reshape = mx.sym.Reshape(linear_out, shape=(dim[2], num_rois, -1))
        # [num_rois, num_fg_classes, dim[2]]
        output = mx.sym.transpose(linear_out_reshape, axes=(1, 2, 0))
        return output, aff_softmax

    def get_symbol_rcnn(self, cfg, is_train=True):

        # config alias for convenient
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)

        # input init
        if is_train:
            data = mx.symbol.Variable(name="data")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            non_gt_index = mx.symbol.Variable(name='nongt_index')
            im_info = mx.sym.Variable(name="im_info")
            rois_0 = mx.symbol.Variable(name='rois_0')
            rois_1 = mx.symbol.Variable(name='rois_1')
            rois_2 = mx.symbol.Variable(name='rois_2')
            rois_3 = mx.symbol.Variable(name='rois_3')
            label = mx.symbol.Variable(name='label')
            bbox_target = mx.symbol.Variable(name='bbox_target')
            bbox_weight = mx.symbol.Variable(name='bbox_weight')
            # reshape input
            rois_0 = mx.symbol.Reshape(data=rois_0, shape=(-1, 5), name='rois_0_reshape')
            rois_1 = mx.symbol.Reshape(data=rois_1, shape=(-1, 5), name='rois_1_reshape')
            rois_2 = mx.symbol.Reshape(data=rois_2, shape=(-1, 5), name='rois_2_reshape')
            rois_3 = mx.symbol.Reshape(data=rois_3, shape=(-1, 5), name='rois_3_reshape')

            label = mx.symbol.Reshape(data=label, shape=(-1,), name='label_reshape')
            bbox_target = mx.symbol.Reshape(data=bbox_target, shape=(-1, 4 * num_reg_classes),
                                            name='bbox_target_reshape')
            bbox_weight = mx.symbol.Reshape(data=bbox_weight, shape=(-1, 4 * num_reg_classes),
                                            name='bbox_weight_reshape')
        else:
            data = mx.sym.Variable(name="data")
            non_gt_index = None
            im_info = mx.symbol.Variable(name='im_info')
            # reshape input
            rois_0 = mx.symbol.Variable(name='rois_0')
            rois_1 = mx.symbol.Variable(name='rois_1')
            rois_2 = mx.symbol.Variable(name='rois_2')
            rois_3 = mx.symbol.Variable(name='rois_3')
            rois_0 = mx.symbol.Reshape(data=rois_0, shape=(-1, 5), name='rois_0_reshape')
            rois_1 = mx.symbol.Reshape(data=rois_1, shape=(-1, 5), name='rois_1_reshape')
            rois_2 = mx.symbol.Reshape(data=rois_2, shape=(-1, 5), name='rois_2_reshape')
            rois_3 = mx.symbol.Reshape(data=rois_3, shape=(-1, 5), name='rois_3_reshape')

        # shared convolutional layers
        fpn_ft64, fpn_ft32, fpn_ft16, fpn_ft8, fpn_ft4 = self.get_resnet_v1_fpn_conv(data)

        roi_pool_ft4 = mx.symbol.ROIPooling(name='roi_pool_ft4', data=fpn_ft4, rois=rois_0, pooled_size=(7, 7),
                                            spatial_scale=1.0 / 4.0)
        roi_pool_ft8 = mx.symbol.ROIPooling(name='roi_pool_ft8', data=fpn_ft8, rois=rois_1, pooled_size=(7, 7),
                                            spatial_scale=1.0 / 8.0)
        roi_pool_ft16 = mx.symbol.ROIPooling(name='roi_pool_ft16', data=fpn_ft16, rois=rois_2, pooled_size=(7, 7),
                                             spatial_scale=1.0 / 16.0)
        roi_pool_ft32 = mx.symbol.ROIPooling(name='roi_pool_ft32', data=fpn_ft32, rois=rois_3, pooled_size=(7, 7),
                                             spatial_scale=1.0 / 32.0)

        roi_pool_concat = mx.symbol.Concat(roi_pool_ft4, roi_pool_ft8, roi_pool_ft16, roi_pool_ft32, dim=0)
        # roi_pool = mx.symbol.take(roi_pool_concat, feat_id)
        rois = mx.symbol.concat(rois_0, rois_1, rois_2, rois_3, dim=0, name='rois')
        sliced_rois = mx.sym.slice_axis(rois, axis=1, begin=1, end=None)
        # [num_rois, nongt_dim, 4]
        position_matrix = self.extract_position_matrix(sliced_rois, non_gt_index=non_gt_index)
        # [num_rois, nongt_dim, 64]
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
        # [64, num_rois, nongt_dim]
        position_embedding_reshape = mx.sym.transpose(position_embedding, axes=(2, 0, 1))
        # [1, 64, num_rois, nongt_dim]
        position_embedding_reshape = mx.sym.expand_dims(position_embedding_reshape, axis=0)

        roi_pool_fc1 = mx.symbol.FullyConnected(name='roi_pool_fc1', data=roi_pool_concat, num_hidden=1024)
        # attention, [num_rois, feat_dim]
        attention_1 = self.attention_module_multi_head(roi_pool_fc1, position_embedding_reshape,
                                                       non_gt_index=non_gt_index, fc_dim=16, feat_dim=1024,
                                                       index=1, group=16,
                                                       dim=(1024, 1024, 1024))
        roi_pool_fc1 = roi_pool_fc1 + attention_1
        roi_pool_relu1 = mx.sym.Activation(data=roi_pool_fc1, act_type='relu', name='roi_pool_relu1')
        roi_pool_fc2 = mx.symbol.FullyConnected(name='roi_pool_fc2', data=roi_pool_relu1, num_hidden=1024)
        # attention, [num_rois, feat_dim]
        attention_2 = self.attention_module_multi_head(roi_pool_fc2, position_embedding_reshape,
                                                       non_gt_index=non_gt_index, fc_dim=16, feat_dim=1024,
                                                       index=2, group=16,
                                                       dim=(1024, 1024, 1024))
        roi_pool_fc2 = roi_pool_fc2 + attention_2
        roi_pool_relu2 = mx.sym.Activation(data=roi_pool_fc2, act_type='relu', name='roi_pool_relu2')

        if is_train:
            cls_score = mx.symbol.FullyConnected(name='cls_score', data=roi_pool_relu2, num_hidden=num_classes)
            bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=roi_pool_relu2, num_hidden=num_reg_classes * 4)

            if cfg.TRAIN.ENABLE_OHEM:
                print "Open OHEM"
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes,
                                                               roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                                normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                                  data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=1.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                print "Close OHEM"
                cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid',
                                                use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                            data=(bbox_pred - bbox_target))

                if cfg.TRAIN.BATCH_ROIS < 0:
                    if cfg.TRAIN.TOP_ROIS < 0:
                        raise ValueError('Please check!')
                    batch_rois_num = cfg.TRAIN.TOP_ROIS
                else:
                    batch_rois_num = cfg.TRAIN.BATCH_ROIS

                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / batch_rois_num)
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                         name='cls_prob_reshape')
            bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                          name='bbox_loss_reshape')

            output_sym_list = [cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)]
        else:
            # classification
            # rois = mx.symbol.Concat(rois_0, rois_1, rois_2, rois_3, dim=0, name='rois')
            cls_score = mx.symbol.FullyConnected(name='cls_score', data=roi_pool_relu2, num_hidden=num_classes)
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            # bounding box regression
            bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=roi_pool_relu2, num_hidden=num_reg_classes * 4)

            # reshape output
            cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                         name='cls_prob_reshape')
            bbox_pred_reshape = mx.symbol.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                                  name='bbox_pred_reshape')

            # group output
            output_sym_list = [rois, cls_prob, bbox_pred_reshape]

        if is_train and (not cfg.TRAIN.LEARN_NMS):
            raise ValueError('config.TRAIN.LEARN_NMS is set to false!')
        elif (not is_train) and (not cfg.TEST.LEARN_NMS):
            self.sym = mx.sym.Group(output_sym_list)
            # print self.sym.list_outputs()
            return self.sym

        ######################### learn nms #########################
        # notice that all implementation of python ops try to leave batch idx support for multi-batch
        # thus, rois are [batch_ind, x_min, y_min, x_max, y_max]
        nms_target_thresh = np.fromstring(cfg.network.NMS_TARGET_THRESH, dtype=float, sep=',')
        num_thresh = len(nms_target_thresh)
        nms_eps = 1e-8
        first_n = cfg.TRAIN.FIRST_N if is_train else cfg.TEST.FIRST_N
        num_fg_classes = num_classes - 1
        # refine bbox
        bbox_means = cfg.TRAIN.BBOX_MEANS if is_train else None
        bbox_stds = cfg.TRAIN.BBOX_STDS if is_train else None

        if is_train:
            # remove gt here
            if non_gt_index is not None:
                cls_score_nongt = mx.sym.take(a=cls_score, indices=non_gt_index)
                bbox_pred_nongt = mx.sym.take(a=bbox_pred, indices=non_gt_index)
                bbox_pred_nongt = mx.sym.BlockGrad(bbox_pred_nongt)
            else:
                cls_score_nongt = cls_score
                bbox_pred_nongt = bbox_pred
                bbox_pred_nongt = mx.sym.BlockGrad(bbox_pred_nongt)

            # remove batch idx and gt roi
            sliced_rois = mx.sym.slice_axis(data=rois, axis=1, begin=1, end=None)
            if non_gt_index is not None:
                sliced_rois = mx.sym.take(a=sliced_rois, indices=non_gt_index)
            # bbox_pred_nobg, [num_rois, 4*(num_reg_classes-1)]
            bbox_pred_nobg = mx.sym.slice_axis(data=bbox_pred_nongt, axis=1, begin=4, end=None)
            # [num_boxes, 4, num_reg_classes-1]
            refined_bbox = NMS_UTILS.refine_bbox(sliced_rois, bbox_pred_nobg, im_info,
                                                 means=bbox_means, stds=bbox_stds
                                                 )
            # softmax cls_score to cls_prob, [num_rois, num_classes]
            cls_prob = mx.sym.softmax(data=cls_score_nongt, axis=-1)
            cls_prob_nobg = mx.sym.slice_axis(cls_prob, axis=1, begin=1, end=None)
            sorted_cls_prob_nobg = mx.sym.sort(data=cls_prob_nobg, axis=0, is_ascend=False)
            # sorted_score, [first_n, num_fg_classes]
            sorted_score = mx.sym.slice_axis(sorted_cls_prob_nobg, axis=0,
                                             begin=0, end=first_n, name='sorted_score')
            # sort by score
            rank_indices = mx.sym.argsort(data=cls_prob_nobg, axis=0, is_ascend=False)
            # first_rank_indices, [first_n, num_fg_classes]
            first_rank_indices = mx.sym.slice_axis(rank_indices, axis=0, begin=0, end=first_n)
            # sorted_bbox, [first_n, num_fg_classes, 4, num_reg_classes-1]
            sorted_bbox = mx.sym.take(a=refined_bbox, indices=first_rank_indices)
            if cfg.CLASS_AGNOSTIC:
                # sorted_bbox, [first_n, num_fg_classes, 4]
                sorted_bbox = mx.sym.Reshape(sorted_bbox, shape=(0, 0, 0), name='sorted_bbox')
            else:
                cls_mask = mx.sym.arange(0, num_fg_classes)
                cls_mask = mx.sym.Reshape(cls_mask, shape=(1, -1, 1))
                cls_mask = mx.sym.broadcast_to(cls_mask, shape=(first_n, 0, 4))
                # sorted_bbox, [first_n, num_fg_classes, 4]
                sorted_bbox = mx.sym.pick(data=sorted_bbox, name='sorted_bbox',
                                          index=cls_mask, axis=3)
            # sorted_bbox = monitor_wrapper(sorted_bbox, 'sorted_bbox')
            # nms_rank_embedding, [first_n, 1024]
            nms_rank_embedding = NMS_UTILS.extract_rank_embedding(first_n, 1024)
            # nms_rank_feat, [first_n, 1024]
            nms_rank_feat = mx.sym.FullyConnected(name='nms_rank', data=nms_rank_embedding, num_hidden=128)
            # nms_position_matrix, [num_fg_classes, first_n, first_n, 4]
            nms_position_matrix = NMS_UTILS.extract_multi_position_matrix(sorted_bbox)
            # roi_feature_embedding, [num_rois, 1024]
            roi_feat_embedding = mx.sym.FullyConnected(
                name='roi_feat_embedding',
                data=roi_pool_relu2,
                num_hidden=128)
            # sorted_roi_feat, [first_n, num_fg_classes, 128]
            sorted_roi_feat = mx.sym.take(a=roi_feat_embedding, indices=first_rank_indices)

            # vectorized nms
            # nms_embedding_feat, [first_n, num_fg_classes, 128]
            nms_embedding_feat = mx.sym.broadcast_add(
                lhs=sorted_roi_feat,
                rhs=mx.sym.expand_dims(nms_rank_feat, axis=1))
            # nms_attention_1, [first_n, num_fg_classes, 1024]
            nms_attention_1, nms_softmax_1 = self.attention_module_nms_multi_head(
                nms_embedding_feat, nms_position_matrix,
                num_rois=first_n, index=1, group=16,
                dim=(1024, 1024, 128), fc_dim=(64, 16), feat_dim=128)
            nms_all_feat_1 = nms_embedding_feat + nms_attention_1
            nms_all_feat_1_relu = mx.sym.Activation(data=nms_all_feat_1, act_type='relu', name='nms_all_feat_1_relu')
            # [first_n * num_fg_classes, 1024]
            nms_all_feat_1_relu_reshape = mx.sym.Reshape(nms_all_feat_1_relu, shape=(-3, -2))
            # logit, [first_n * num_fg_classes, num_thresh]
            nms_conditional_logit = mx.sym.FullyConnected(name='nms_logit',
                                                          data=nms_all_feat_1_relu_reshape,
                                                          num_hidden=num_thresh)
            # logit_reshape, [first_n, num_fg_classes, num_thresh]
            nms_conditional_logit_reshape = mx.sym.Reshape(nms_conditional_logit,
                                                           shape=(first_n, num_fg_classes, num_thresh))
            nms_conditional_score = mx.sym.Activation(data=nms_conditional_logit_reshape,
                                                      act_type='sigmoid', name='nms_conditional_score')
            sorted_score_reshape = mx.sym.expand_dims(sorted_score, axis=2)
            # sorted_score_reshape = mx.sym.BlockGrad(sorted_score_reshape)
            nms_multi_score = mx.sym.broadcast_mul(lhs=sorted_score_reshape, rhs=nms_conditional_score)
        else:
            nms_rank_weight = mx.sym.var('nms_rank_weight', shape=(128, 1024), dtype=np.float32)
            nms_rank_bias = mx.sym.var('nms_rank_bias', shape=(128,), dtype=np.float32)
            roi_feat_embedding_weight = mx.sym.var('roi_feat_embedding_weight', shape=(128, 1024), dtype=np.float32)
            roi_feat_embedding_bias = mx.sym.var('roi_feat_embedding_bias', shape=(128,), dtype=np.float32)
            nms_pair_pos_fc1_1_weight = mx.sym.var('nms_pair_pos_fc1_1_weight', shape=(16, 64), dtype=np.float32)
            nms_pair_pos_fc1_1_bias = mx.sym.var('nms_pair_pos_fc1_1_bias', shape=(16,), dtype=np.float32)
            nms_query_1_weight = mx.sym.var('nms_query_1_weight', shape=(1024, 128), dtype=np.float32)
            nms_query_1_bias = mx.sym.var('nms_query_1_bias', shape=(1024,), dtype=np.float32)
            nms_key_1_weight = mx.sym.var('nms_key_1_weight', shape=(1024, 128), dtype=np.float32)
            nms_key_1_bias = mx.sym.var('nms_key_1_bias', shape=(1024,), dtype=np.float32)
            nms_linear_out_1_weight = mx.sym.var('nms_linear_out_1_weight', shape=(128, 128, 1, 1), dtype=np.float32)
            nms_linear_out_1_bias = mx.sym.var('nms_linear_out_1_bias', shape=(128,), dtype=np.float32)
            nms_logit_weight = mx.sym.var('nms_logit_weight', shape=(5, 128), dtype=np.float32)
            nms_logit_bias = mx.sym.var('nms_logit_bias', shape=(5,), dtype=np.float32)


            learn_nms_params = {
                'cls_score': cls_score, 
                'bbox_pred': bbox_pred, 
                'rois': rois, 
                'im_info': im_info, 
                'fc_all_2_relu': roi_pool_relu2,
                'nms_rank_weight': nms_rank_weight, 
                'nms_rank_bias': nms_rank_bias, 
                'roi_feat_embedding_weight': roi_feat_embedding_weight,
                'roi_feat_embedding_bias': roi_feat_embedding_bias, 
                'nms_pair_pos_fc1_1_weight': nms_pair_pos_fc1_1_weight, 
                'nms_pair_pos_fc1_1_bias': nms_pair_pos_fc1_1_bias, 
                'nms_query_1_weight': nms_query_1_weight, 
                'nms_query_1_bias': nms_query_1_bias, 
                'nms_key_1_weight': nms_key_1_weight, 
                'nms_key_1_bias': nms_key_1_bias,
                'nms_linear_out_1_weight': nms_linear_out_1_weight, 
                'nms_linear_out_1_bias': nms_linear_out_1_bias, 
                'nms_logit_weight': nms_logit_weight, 
                'nms_logit_bias': nms_logit_bias,
                'op_type': 'learn_nms', 
                'name': 'learn_nms',
                'num_fg_classes': num_fg_classes, 
                'bbox_means': bbox_means, 
                'bbox_stds': bbox_stds, 
                'first_n':first_n, 
                'class_agnostic': cfg.CLASS_AGNOSTIC, 
                'num_thresh': num_thresh, 
                'class_thresh': cfg.TEST.LEARN_NMS_CLASS_SCORE_TH, 
                'nongt_dim': None, 
                'has_non_gt_index':(non_gt_index is not None)
            }
            if non_gt_index is not None:
                learn_nms_params['non_gt_index'] = non_gt_index
            nms_multi_score, sorted_bbox, sorted_score = mx.sym.Custom(**learn_nms_params)

        if is_train:
            nms_multi_target = mx.sym.Custom(bbox=sorted_bbox, gt_bbox=gt_boxes, score=sorted_score,
                                             op_type='nms_multi_target', target_thresh=nms_target_thresh)
            nms_pos_loss = - mx.sym.broadcast_mul(lhs=nms_multi_target,
                                                  rhs=mx.sym.log(data=(nms_multi_score + nms_eps)))
            nms_neg_loss = - mx.sym.broadcast_mul(lhs=(1.0 - nms_multi_target),
                                                  rhs=mx.sym.log(data=(1.0 - nms_multi_score + nms_eps)))
            normalizer = first_n * num_thresh
            nms_pos_loss = cfg.TRAIN.nms_loss_scale * nms_pos_loss / normalizer
            nms_neg_loss = cfg.TRAIN.nms_loss_scale * nms_neg_loss / normalizer
            ##########################  additional output!  ##########################
            output_sym_list.append(mx.sym.BlockGrad(nms_multi_target, name='nms_multi_target_block'))
            output_sym_list.append(mx.sym.BlockGrad(nms_conditional_score, name='nms_conditional_score_block'))
            output_sym_list.append(mx.sym.MakeLoss(name='nms_pos_loss', data=nms_pos_loss,
                                                   grad_scale=cfg.TRAIN.nms_pos_scale))
            output_sym_list.append(mx.sym.MakeLoss(name='nms_neg_loss', data=nms_neg_loss))
        else:
            if cfg.TEST.MERGE_METHOD == -1:
                nms_final_score = mx.sym.mean(data=nms_multi_score, axis=2, name='nms_final_score')
            elif cfg.TEST.MERGE_METHOD == -2:
                nms_final_score = mx.sym.max(data=nms_multi_score, axis=2, name='nms_final_score')
            elif 0 <= cfg.TEST.MERGE_METHOD < num_thresh:
                idx = cfg.TEST.MERGE_METHOD
                nms_final_score = mx.sym.slice_axis(data=nms_multi_score, axis=2, begin=idx, end=idx + 1)
                nms_final_score = mx.sym.Reshape(nms_final_score, shape=(0, 0), name='nms_final_score')
            else:
                raise NotImplementedError('Unknown merge method %s.' % cfg.TEST.MERGE_METHOD)
            output_sym_list.append(sorted_bbox)
            output_sym_list.append(sorted_score)
            output_sym_list.append(nms_final_score)

        self.sym = mx.sym.Group(output_sym_list)
        # print self.sym.list_outputs()
        return self.sym

    def init_weight_attention_nms_multi_head(self, cfg, arg_params, aux_params, index=1):
        arg_params['nms_pair_pos_fc1_' + str(index) + '_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_pair_pos_fc1_' + str(index) + '_weight'])
        arg_params['nms_pair_pos_fc1_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['nms_pair_pos_fc1_' + str(index) + '_bias'])
        arg_params['nms_query_' + str(index) + '_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_query_' + str(index) + '_weight'])
        arg_params['nms_query_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['nms_query_' + str(index) + '_bias'])
        arg_params['nms_key_' + str(index) + '_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_key_' + str(index) + '_weight'])
        arg_params['nms_key_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['nms_key_' + str(index) + '_bias'])
        arg_params['nms_linear_out_' + str(index) + '_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_linear_out_' + str(index) + '_weight'])
        arg_params['nms_linear_out_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['nms_linear_out_' + str(index) + '_bias'])

    def init_weight_nms(self, cfg, arg_params,aux_params):
        arg_params['nms_rank_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_rank_weight'])
        arg_params['nms_rank_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['nms_rank_bias'])
        arg_params['roi_feat_embedding_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['roi_feat_embedding_weight'])
        arg_params['roi_feat_embedding_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['roi_feat_embedding_bias'])
        self.init_weight_attention_nms_multi_head(cfg, arg_params, aux_params, index=1)
        arg_params['nms_logit_weight'] = mx.random.normal(
            0, 0.01, shape=self.arg_shape_dict['nms_logit_weight'])
        arg_params['nms_logit_bias'] = mx.nd.full(shape=self.arg_shape_dict['nms_logit_bias'], val=-3.0)

    def init_weight_attention_multi_head(self, cfg, arg_params, aux_params, index=1):
        arg_params['pair_pos_fc1_' + str(index) + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'pair_pos_fc1_' + str(index) + '_weight'])
        arg_params['pair_pos_fc1_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['pair_pos_fc1_' + str(index) + '_bias'])
        # batch mode
        arg_params['query_' + str(index) + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'query_' + str(index) + '_weight'])
        arg_params['query_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['query_' + str(index) + '_bias'])
        arg_params['key_' + str(index) + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'key_' + str(index) + '_weight'])
        arg_params['key_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['key_' + str(index) + '_bias'])
        arg_params['linear_out_' + str(index) + '_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[
            'linear_out_' + str(index) + '_weight'])
        arg_params['linear_out_' + str(index) + '_bias'] = mx.nd.zeros(
            shape=self.arg_shape_dict['linear_out_' + str(index) + '_bias'])

    def init_fpn_weight(self, cfg, arg_params, aux_params, has_ft64=True):
        arg_params['fpn_ft32_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_ft32_1x1_weight'])
        arg_params['fpn_ft32_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_ft32_1x1_bias'])
        arg_params['fpn_ft16_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_ft16_1x1_weight'])
        arg_params['fpn_ft16_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_ft16_1x1_bias'])
        arg_params['fpn_ft8_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_ft8_1x1_weight'])
        arg_params['fpn_ft8_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_ft8_1x1_bias'])
        arg_params['fpn_ft4_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_ft4_1x1_weight'])
        arg_params['fpn_ft4_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_ft4_1x1_bias'])

        if has_ft64:
            arg_params['fpn_ft64_3x3_weight'] = mx.random.normal(0, 0.01,
                                                                 shape=self.arg_shape_dict['fpn_ft64_3x3_weight'])
            arg_params['fpn_ft64_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_ft64_3x3_bias'])
        arg_params['fpn_ft32_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_ft32_3x3_weight'])
        arg_params['fpn_ft32_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_ft32_3x3_bias'])
        arg_params['fpn_ft16_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_ft16_3x3_weight'])
        arg_params['fpn_ft16_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_ft16_3x3_bias'])
        arg_params['fpn_ft8_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_ft8_3x3_weight'])
        arg_params['fpn_ft8_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_ft8_3x3_bias'])
        arg_params['fpn_ft4_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_ft4_3x3_weight'])
        arg_params['fpn_ft4_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_ft4_3x3_bias'])

    def init_weight(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])
        arg_params['rfcn_cls_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_cls_weight'])
        arg_params['rfcn_cls_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_cls_bias'])
        arg_params['rfcn_bbox_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rfcn_bbox_weight'])
        arg_params['rfcn_bbox_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rfcn_bbox_bias'])
        self.init_fpn_weight(cfg, arg_params, aux_params)

    def init_weight_rpn(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01,
                                                              shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])
        self.init_fpn_weight(cfg, arg_params, aux_params)

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        if cfg.TRAIN.JOINT_TRAINING:
            arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
            arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
            arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
            arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

            # arg_params['offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_weight'])
            # arg_params['offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_bias'])

            arg_params['roi_pool_fc1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['roi_pool_fc1_weight'])
            arg_params['roi_pool_fc1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['roi_pool_fc1_bias'])
            arg_params['roi_pool_fc2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['roi_pool_fc2_weight'])
            arg_params['roi_pool_fc2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['roi_pool_fc2_bias'])
            self.init_fpn_weight(cfg, arg_params, aux_params, has_ft64=False)
            for idx in range(2):
                self.init_weight_attention_multi_head(cfg, arg_params, aux_params, index=idx+1)

        # init learn nms
        self.init_weight_nms(cfg, arg_params, aux_params)
