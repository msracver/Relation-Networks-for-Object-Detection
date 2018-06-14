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

import mxnet as mx
import numpy as np


def get_rpn_names():
    pred = ['rpn_cls_prob', 'rpn_bbox_loss']
    label = ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']
    return pred, label


def get_rcnn_names(cfg):
    pred = ['rcnn_cls_prob', 'rcnn_bbox_loss']
    label = ['rcnn_label', 'rcnn_bbox_target', 'rcnn_bbox_weight']
    if cfg.TRAIN.ENABLE_OHEM or cfg.TRAIN.END2END:
        pred.append('rcnn_label')
    if cfg.TRAIN.END2END:
        rpn_pred, rpn_label = get_rpn_names()
        pred = rpn_pred + pred
        label = rpn_label
    return pred, label


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # pred (b, c, p) or (b, c, h, w)
        pred_label = mx.ndarray.argmax_channel(pred).asnumpy().astype('int32')
        pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNAccMetric, self).__init__('RCNNAcc')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred_label = pred.asnumpy().reshape(-1, last_dim).argmax(axis=1).astype('int32')
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RPNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNLogLossMetric, self).__init__('RPNLogLoss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        pred = preds[self.pred.index('rpn_cls_prob')]
        label = labels[self.label.index('rpn_label')]

        # label (b, p)
        label = label.asnumpy().astype('int32').reshape((-1))
        # pred (b, c, p) or (b, c, h, w) --> (b, p, c) --> (b*p, c)
        pred = pred.asnumpy().reshape((pred.shape[0], pred.shape[1], -1)).transpose((0, 2, 1))
        pred = pred.reshape((label.shape[0], -1))

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNLogLossMetric, self).__init__('RCNNLogLoss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        pred = preds[self.pred.index('rcnn_cls_prob')]
        if self.ohem or self.e2e:
            label = preds[self.pred.index('rcnn_label')]
        else:
            label = labels[self.label.index('rcnn_label')]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')
        self.pred, self.label = get_rpn_names()

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rpn_bbox_loss')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        label = labels[self.label.index('rpn_label')].asnumpy()
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')
        self.e2e = cfg.TRAIN.END2END
        self.ohem = cfg.TRAIN.ENABLE_OHEM
        self.pred, self.label = get_rcnn_names(cfg)

    def update(self, labels, preds):
        bbox_loss = preds[self.pred.index('rcnn_bbox_loss')].asnumpy()
        if self.ohem:
            label = preds[self.pred.index('rcnn_label')].asnumpy()
        else:
            if self.e2e:
                label = preds[self.pred.index('rcnn_label')].asnumpy()
            else:
                label = labels[self.label.index('rcnn_label')].asnumpy()

        # calculate num_inst (average on those kept anchors)
        num_inst = np.sum(label != -1)

        self.sum_metric += np.sum(bbox_loss)
        self.num_inst += num_inst


class NMSLossMetric(mx.metric.EvalMetric):
    def __init__(self, cfg, name):
        assert cfg.TRAIN.LEARN_NMS, 'config set learn nms to be false'
        assert name in ['pos', 'neg'], 'only for nms_pos/neg_loss'
        super(NMSLossMetric, self).__init__('NMSLoss_' + name)
        # self._num_fg_classes = cfg.dataset.NUM_CLASSES - 1
        self._offset = ['pos', 'neg'].index(name)

    def update(self, labels, preds):
        # for x in preds:
        #     nms_loss_list.append(x.asnumpy())
        # if self._debug:
        #     nms_multi_target = preds[-4].asnumpy()
        #     if self._offset == 0:
        #         print self.name, 'pos:', np.sum(nms_multi_target[1:, :, :])
        #     else:
        #         print self.name, 'neg:', np.size(nms_multi_target[1:, :, :]) - np.sum(nms_multi_target[1:, :, :])

        nms_loss = preds[-2 + self._offset].asnumpy()

        self.sum_metric += np.sum(nms_loss)
        self.num_inst += 1


# v0.10.0 support update_dict, but v0.9.5 does not support
class NMSAccMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        assert cfg.TRAIN.LEARN_NMS, 'config set learn nms to be false'
        self._suffixes=['pos', 'neg']
        super(NMSAccMetric, self).__init__('NMSAcc')

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = [0, 0]
        self.sum_metric = [0.0, 0.0]

    def get(self):
        name = []
        value = []
        for idx, num_inst in enumerate(self.num_inst):
            name.append(self.name + '_' +self._suffixes[idx])
            if num_inst == 0:
                value.append(float('nan'))
            else:
                value.append(self.sum_metric[idx] / self.num_inst[idx])
        return name, value

    def update(self, labels, preds):
        nms_multi_target = preds[-4].asnumpy()
        nms_conditional_score = preds[-3].asnumpy()

        # pos
        valid_mask = nms_multi_target > 0.5
        valid_score = (nms_conditional_score > 0.5)
        num_inst = np.sum(valid_mask)
        num_true = np.sum(valid_mask * valid_score)
        self.sum_metric[0] += num_true
        self.num_inst[0] += num_inst
        # neg
        valid_mask = nms_multi_target < 0.5
        valid_score = (nms_conditional_score < 0.5)
        num_inst = np.sum(valid_mask)
        num_true = np.sum(valid_mask * valid_score)
        self.sum_metric[1] += num_true
        self.num_inst[1] += num_inst


class NMSAccValidMetric(mx.metric.EvalMetric):
    def __init__(self, cfg):
        assert cfg.TRAIN.LEARN_NMS, 'config set learn nms to be false'
        assert cfg.TRAIN.INSTANCE_WEIGHT, 'config set instance weight to be false'
        self._suffixes=['pos', 'neg']
        super(NMSAccValidMetric, self).__init__('NMSAccValid')

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = [0, 0]
        self.sum_metric = [0.0, 0.0]

    def get(self):
        name = []
        value = []
        for idx, num_inst in enumerate(self.num_inst):
            name.append(self.name + '_' +self._suffixes[idx])
            if num_inst == 0:
                value.append(float('nan'))
            else:
                value.append(self.sum_metric[idx] / self.num_inst[idx])
        return name, value

    def update(self, labels, preds):
        nms_multi_target = preds[-4].asnumpy()
        nms_conditional_score = preds[-3].asnumpy()
        instance_weight = preds[-5].asnumpy()
        instance_mask = (instance_weight > 1e-8)

        # pos
        valid_mask = nms_multi_target > 0.5
        valid_mask = valid_mask * instance_mask
        valid_score = (nms_conditional_score > 0.5)
        num_inst = np.sum(valid_mask)
        num_true = np.sum(valid_mask * valid_score)
        self.sum_metric[0] += num_true
        self.num_inst[0] += num_inst
        # neg
        valid_mask = nms_multi_target < 0.5
        valid_mask = valid_mask * instance_mask
        valid_score = (nms_conditional_score < 0.5)
        num_inst = np.sum(valid_mask)
        num_true = np.sum(valid_mask * valid_score)
        self.sum_metric[1] += num_true
        self.num_inst[1] += num_inst
