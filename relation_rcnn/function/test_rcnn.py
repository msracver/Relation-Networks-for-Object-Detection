# --------------------------------------------------------
# Relation Networks for Object Detection
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Jiayuan Gu, Dazhi Cheng, Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------


import argparse
import pprint
import logging
import time
import os
import mxnet as mx

from symbols import *
from dataset import *
from core.loader import TestLoader
from core.tester import Predictor, pred_eval
from utils.load_model import load_param


def test_rcnn(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        roidb = imdb.gt_roidb()
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)
        rpn_path = cfg.dataset.proposal_cache
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path, rpn_path=rpn_path)
        gt_roidb = imdb.gt_roidb()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb, top_roi=cfg.TEST.TOP_ROIS)

    # get test data iter
    test_data = TestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    #sym_instance.infer_shape(data_shape_dict)

    #sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    #max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    max_height = max([v[0] for v in cfg.SCALES])
    max_width = max([v[1] for v in cfg.SCALES])
    if cfg.network.IMAGE_STRIDE > 0:
        max_height = max_height + cfg.network.IMAGE_STRIDE - max_height%cfg.network.IMAGE_STRIDE
        max_width = max_width + cfg.network.IMAGE_STRIDE - max_width % cfg.network.IMAGE_STRIDE

    max_data_shape = [('data', (cfg.TRAIN.BATCH_IMAGES, 3, max_height, max_width))]

    if not has_rpn:
        #max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))
        if cfg.network.ROIDispatch:
            max_data_shape.append(('rois_0', (1, cfg.TEST.PROPOSAL_POST_NMS_TOP_N/4, 5)))
            max_data_shape.append(('rois_1', (1, cfg.TEST.PROPOSAL_POST_NMS_TOP_N/4, 5)))
            max_data_shape.append(('rois_2', (1, cfg.TEST.PROPOSAL_POST_NMS_TOP_N/4, 5)))
            max_data_shape.append(('rois_3', (1, cfg.TEST.PROPOSAL_POST_NMS_TOP_N/4, 5)))

    max_data_shape = [max_data_shape]
    # create predictor
    #test_data.provide_label
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start detection
    pred_eval(predictor, test_data, imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)

