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


"""
Fast R-CNN:
data =
    {'data': [num_images, c, h, w],
    'rois': [num_rois, 5]}
label =
    {'label': [num_rois],
    'bbox_target': [num_rois, 4 * num_classes],
    'bbox_weight': [num_rois, 4 * num_classes]}
roidb extended format [image_index]
    ['image', 'height', 'width', 'flipped',
     'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""

import numpy as np
import numpy.random as npr

from utils.image import get_image, tensor_vstack
from bbox.bbox_transform import bbox_overlaps, bbox_transform
from bbox.bbox_regression import expand_bbox_regression_targets


def get_rcnn_testbatch(roidb, cfg):
    """
    return a dict of testbatch
    :param roidb: ['image', 'flipped'] + ['boxes']
    :return: data, label, im_info
    """
    # assert len(roidb) == 1, 'Single batch only'
    imgs, roidb = get_image(roidb, cfg)
    im_array = imgs
    im_info = [np.array([roidb[i]['im_info']], dtype=np.float32) for i in range(len(roidb))]

    im_rois = [roidb[i]['boxes'] for i in range(len(roidb))]

    if cfg.network.ROIDispatch:
        data = []
        for i in range(len(im_rois)):
            w = im_rois[i][:, 2] - im_rois[i][:, 0] + 1
            h = im_rois[i][:, 3] - im_rois[i][:, 1] + 1
            feat_id = np.clip(np.floor(2 + np.log2(np.sqrt(w * h) / 224)), 0, 3).astype(int)

            rois_0 = im_rois[i][np.where(feat_id == 0)]
            if len(rois_0) == 0:
                rois_0 = np.zeros((1, 4))
            rois_1 = im_rois[i][np.where(feat_id == 1)]
            if len(rois_1) == 0:
                rois_1 = np.zeros((1, 4))
            rois_2 = im_rois[i][np.where(feat_id == 2)]
            if len(rois_2) == 0:
                rois_2 = np.zeros((1, 4))
            rois_3 = im_rois[i][np.where(feat_id == 3)]
            if len(rois_3) == 0:
                rois_3 = np.zeros((1, 4))
            # stack batch index
            data.append({'data': im_array[i],
                         'rois_0': np.hstack((0 * np.ones((rois_0.shape[0], 1)), rois_0)),
                         'rois_1': np.hstack((0 * np.ones((rois_1.shape[0], 1)), rois_1)),
                         'rois_2': np.hstack((0 * np.ones((rois_2.shape[0], 1)), rois_2)),
                         'rois_3': np.hstack((0 * np.ones((rois_3.shape[0], 1)), rois_3))})
            if cfg.TEST.LEARN_NMS:
                data[-1]['im_info'] = im_info[i]
    else:
        rois = im_rois
        rois_array = [np.hstack((0 * np.ones((rois[i].shape[0], 1)), rois[i])) for i in range(len(rois))]

        data = []
        for i in range(len(roidb)):
            data.append({'data': im_array[i],
                         'rois': rois_array[i]})
            if cfg.TEST.LEARN_NMS:
                data[-1]['im_info'] = im_info[i]

    label = {}

    return data, label, im_info


def get_rcnn_batch(roidb, cfg):
    """
    return a dict of multiple images
    :param roidb: a list of dict, whose length controls batch size
    ['images', 'flipped'] + ['gt_boxes', 'boxes', 'gt_overlap'] => ['bbox_targets']
    :return: data, label
    """
    num_images = len(roidb)
    imgs, roidb = get_image(roidb, cfg)
    im_array = tensor_vstack(imgs)

    assert cfg.TRAIN.BATCH_ROIS == -1 or cfg.TRAIN.BATCH_ROIS % cfg.TRAIN.BATCH_IMAGES == 0, \
        'BATCHIMAGES {} must divide BATCH_ROIS {}'.format(cfg.TRAIN.BATCH_IMAGES, cfg.TRAIN.BATCH_ROIS)

    if cfg.TRAIN.BATCH_ROIS == -1:
        rois_per_image = np.sum([iroidb['boxes'].shape[0] for iroidb in roidb])
        fg_rois_per_image = rois_per_image
    else:
        rois_per_image = cfg.TRAIN.BATCH_ROIS / cfg.TRAIN.BATCH_IMAGES
        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(int)

    if cfg.network.ROIDispatch:
        rois_array_0 = list()
        rois_array_1 = list()
        rois_array_2 = list()
        rois_array_3 = list()
    else:
        rois_array = list()

    gt_labels_array = list()
    labels_array = list()
    bbox_targets_array = list()
    bbox_weights_array = list()

    for im_i in range(num_images):
        roi_rec = roidb[im_i]

        # infer num_classes from gt_overlaps
        num_classes = roi_rec['gt_overlaps'].shape[1]

        # label = class RoI has max overlap with
        rois = roi_rec['boxes']
        labels = roi_rec['max_classes']
        overlaps = roi_rec['max_overlaps']
        bbox_targets = roi_rec['bbox_targets']
        gt_lables = roi_rec['is_gt']

        if cfg.TRAIN.BATCH_ROIS == -1:
            im_rois, labels_t, bbox_targets, bbox_weights = \
                sample_rois_v2(rois, num_classes, cfg, labels=labels, overlaps=overlaps, bbox_targets=bbox_targets, gt_boxes=None)

            assert np.abs(im_rois - rois).max() < 1e-3
            assert np.abs(labels_t - labels).max() < 1e-3
        else:
            im_rois, labels, bbox_targets, bbox_weights, gt_lables =  \
                sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                            labels, overlaps, bbox_targets, gt_lables=gt_lables)

        # project im_rois
        # do not round roi
        if cfg.network.ROIDispatch:
            w = im_rois[:, 2] - im_rois[:, 0] + 1
            h = im_rois[:, 3] - im_rois[:, 1] + 1
            feat_id = np.clip(np.floor(2 + np.log2(np.sqrt(w * h) / 224)), 0, 3).astype(int)

            rois_0_idx = np.where(feat_id == 0)[0]
            rois_0 = im_rois[rois_0_idx]
            if len(rois_0) == 0:
                rois_0 = np.zeros((1, 4))
                label_0 = -np.ones((1,))
                gt_label_0 = -np.ones((1,))
                bbox_targets_0 = np.zeros((1, bbox_targets.shape[1]))
                bbox_weights_0 = np.zeros((1, bbox_weights.shape[1]))
            else:
                label_0 = labels[rois_0_idx]
                gt_label_0 = gt_lables[rois_0_idx]
                bbox_targets_0 = bbox_targets[rois_0_idx]
                bbox_weights_0 = bbox_weights[rois_0_idx]

            rois_1_idx = np.where(feat_id == 1)[0]
            rois_1 = im_rois[rois_1_idx]
            if len(rois_1) == 0:
                rois_1 = np.zeros((1, 4))
                label_1 = -np.ones((1,))
                gt_label_1 = -np.ones((1,))
                bbox_targets_1 = np.zeros((1, bbox_targets.shape[1]))
                bbox_weights_1 = np.zeros((1, bbox_weights.shape[1]))
            else:
                label_1 = labels[rois_1_idx]
                gt_label_1 = gt_lables[rois_1_idx]
                bbox_targets_1 = bbox_targets[rois_1_idx]
                bbox_weights_1 = bbox_weights[rois_1_idx]

            rois_2_idx = np.where(feat_id == 2)
            rois_2 = im_rois[rois_2_idx]
            if len(rois_2) == 0:
                rois_2 = np.zeros((1, 4))
                label_2 = -np.ones((1,))
                gt_label_2 = -np.ones((1,))
                bbox_targets_2 = np.zeros((1, bbox_targets.shape[1]))
                bbox_weights_2 = np.zeros((1, bbox_weights.shape[1]))
            else:
                label_2 = labels[rois_2_idx]
                gt_label_2 = gt_lables[rois_2_idx]
                bbox_targets_2 = bbox_targets[rois_2_idx]
                bbox_weights_2 = bbox_weights[rois_2_idx]

            rois_3_idx = np.where(feat_id == 3)
            rois_3 = im_rois[rois_3_idx]
            if len(rois_3) == 0:
                rois_3 = np.zeros((1, 4))
                label_3 = -np.ones((1,))
                gt_label_3 = -np.ones((1,))
                bbox_targets_3 = np.zeros((1, bbox_targets.shape[1]))
                bbox_weights_3 = np.zeros((1, bbox_weights.shape[1]))
            else:
                label_3 = labels[rois_3_idx]
                gt_label_3 = gt_lables[rois_3_idx]
                bbox_targets_3 = bbox_targets[rois_3_idx]
                bbox_weights_3 = bbox_weights[rois_3_idx]

            # stack batch index
            rois_array_0.append(np.hstack((im_i * np.ones((rois_0.shape[0], 1)), rois_0)))
            rois_array_1.append(np.hstack((im_i * np.ones((rois_1.shape[0], 1)), rois_1)))
            rois_array_2.append(np.hstack((im_i * np.ones((rois_2.shape[0], 1)), rois_2)))
            rois_array_3.append(np.hstack((im_i * np.ones((rois_3.shape[0], 1)), rois_3)))

            labels = np.concatenate([label_0, label_1, label_2, label_3], axis=0)
            gt_lables = np.concatenate([gt_label_0, gt_label_1, gt_label_2, gt_label_3], axis=0)
            bbox_targets = np.concatenate([bbox_targets_0, bbox_targets_1, bbox_targets_2, bbox_targets_3], axis=0)
            bbox_weights = np.concatenate([bbox_weights_0, bbox_weights_1, bbox_weights_2, bbox_weights_3], axis=0)
        else:
            rois = im_rois
            batch_index = im_i * np.ones((rois.shape[0], 1))
            rois_array_this_image = np.hstack((batch_index, rois))
            rois_array.append(rois_array_this_image)

        # add labels
        gt_labels_array.append(gt_lables)
        labels_array.append(labels)
        bbox_targets_array.append(bbox_targets)
        bbox_weights_array.append(bbox_weights)

    gt_labels_array = np.array(gt_labels_array)
    nongt_index_array = np.where(gt_labels_array == 0)[1]
    labels_array = np.array(labels_array)
    bbox_targets_array = np.array(bbox_targets_array)
    bbox_weights_array = np.array(bbox_weights_array)

    if cfg.network.USE_NONGT_INDEX:

        label = {'label': labels_array,
                 'nongt_index': nongt_index_array,
                 'bbox_target': bbox_targets_array,
                 'bbox_weight': bbox_weights_array}

    else:
        label = {'label': labels_array,
                 'bbox_target': bbox_targets_array,
                 'bbox_weight': bbox_weights_array}

    if cfg.network.ROIDispatch:
        rois_array_0 = np.array(rois_array_0)
        rois_array_1 = np.array(rois_array_1)
        rois_array_2 = np.array(rois_array_2)
        rois_array_3 = np.array(rois_array_3)
        # rois_concate = np.concatenate((rois_array_0, rois_array_1, rois_array_2, rois_array_3), axis=1)
        # gt_rois_t = rois_concate[:, gt_labels_array[0,:] > 0]
        data = {'data': im_array,
                'rois_0': rois_array_0,
                'rois_1': rois_array_1,
                'rois_2': rois_array_2,
                'rois_3': rois_array_3}
    else:
        rois_array = np.array(rois_array)
        data = {'data': im_array,
                'rois': rois_array}

    if cfg.TRAIN.LEARN_NMS:
        # im info
        im_info = np.array([roidb[0]['im_info']], dtype=np.float32)
        # gt_boxes
        if roidb[0]['gt_classes'].size > 0:
            gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
            gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
            gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
            gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        else:
            gt_boxes = np.empty((0, 5), dtype=np.float32)
        data['im_info'] = im_info
        data['gt_boxes'] = gt_boxes

    return data, label


def sample_rois_v2(rois, num_classes, cfg,
                   labels=None, overlaps=None, bbox_targets=None, gt_boxes=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    if labels is None:
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]

    # set labels of bg_rois to be 0
    bg_ind = np.where(overlaps < cfg.TRAIN.BG_THRESH_HI)[0]
    labels[bg_ind] = 0

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets
    else:
        targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment, :4])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes, cfg)

    return rois, labels, bbox_targets, bbox_weights



def sample_rois(rois, fg_rois_per_image, rois_per_image, num_classes, cfg,
                labels=None, overlaps=None, bbox_targets=None, gt_boxes=None, gt_lables=None):
    """
    generate random sample of ROIs comprising foreground and background examples
    :param rois: all_rois [n, 4]; e2e: [n, 5] with batch_index
    :param fg_rois_per_image: foreground roi number
    :param rois_per_image: total roi number
    :param num_classes: number of classes
    :param labels: maybe precomputed
    :param overlaps: maybe precomputed (max_overlaps)
    :param bbox_targets: maybe precomputed
    :param gt_boxes: optional for e2e [n, 5] (x1, y1, x2, y2, cls)
    :return: (labels, rois, bbox_targets, bbox_weights)
    """
    if labels is None:
        overlaps = bbox_overlaps(rois[:, 1:].astype(np.float), gt_boxes[:, :4].astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) & (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    # pad more to ensure a fixed minibatch size
    while keep_indexes.shape[0] < rois_per_image:
        gap = np.minimum(len(rois), rois_per_image - keep_indexes.shape[0])
        gap_indexes = npr.choice(range(len(rois)), size=gap, replace=False)
        keep_indexes = np.append(keep_indexes, gap_indexes)

    # select gt_labels
    gt_lables = gt_lables[keep_indexes]
    # select labels
    labels = labels[keep_indexes]
    # set labels of bg_rois to be 0
    bg_ind = np.where(overlaps[keep_indexes] < cfg.TRAIN.BG_THRESH_HI)[0]
    labels[bg_ind] = 0
    rois = rois[keep_indexes]

    # load or compute bbox_target
    if bbox_targets is not None:
        bbox_target_data = bbox_targets[keep_indexes, :]
    else:
        targets = bbox_transform(rois[:, 1:], gt_boxes[gt_assignment[keep_indexes], :4])
        if cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            targets = ((targets - np.array(cfg.TRAIN.BBOX_MEANS))
                       / np.array(cfg.TRAIN.BBOX_STDS))
        bbox_target_data = np.hstack((labels[:, np.newaxis], targets))

    bbox_targets, bbox_weights = \
        expand_bbox_regression_targets(bbox_target_data, num_classes, cfg)

    return rois, labels, bbox_targets, bbox_weights, gt_lables

