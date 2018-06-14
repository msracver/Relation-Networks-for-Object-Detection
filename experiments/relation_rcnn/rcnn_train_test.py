# --------------------------------------------------------
# Relation Networks for Object Detection
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiayuan Gu, Dazhi Cheng, Guodong Zhang
# --------------------------------------------------------

import cv2
import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
os.environ['MXNET_GPU_MEM_POOL_RESERVE'] = '10'
os.environ['MXNET_BACKWARD_DO_MIRROR'] = '1'
# os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'relation_rcnn'))

import train_rcnn
import test

if __name__ == "__main__":
    train_rcnn.main()
    test.main()




