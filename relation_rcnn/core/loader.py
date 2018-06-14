# --------------------------------------------------------
# Relation Networks for Object Detection
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import numpy as np
import mxnet as mx
from mxnet.executor_manager import _split_input_slice

from config.config import config
from utils.image import tensor_vstack
from rpn.rpn import get_rpn_testbatch, get_rpn_batch, assign_anchor
from rcnn import get_rcnn_testbatch, get_rcnn_batch
import threading


class TestLoader(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size=1, shuffle=False,
                 has_rpn=False):
        super(TestLoader, self).__init__()

        # save parameters as properties
        self.cfg = config
        self.roidb = roidb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.has_rpn = has_rpn

        # infer properties from roidb
        self.size = len(self.roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        if has_rpn:
            self.data_name = ['data', 'im_info']
        else:
            if self.cfg.network.ROIDispatch:
                self.data_name = ['data', 'rois_0', 'rois_1', 'rois_2', 'rois_3']
            else:
                self.data_name = ['data', 'rois']
            if self.cfg.TEST.LEARN_NMS:
                self.data_name.append('im_info')
        self.label_name = None

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.data = None
        self.label = []
        self.im_info = None

        self.lock = threading.Lock()
        self.lock_data = threading.RLock()

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch()

    def provide_data_func(self, data=None):
        self.lock_data.acquire()
        if data is None:
            data = self.data
        shape = [[(k, v.shape) for k, v in zip(self.data_name, data[i])] for i in xrange(len(data))]
        self.lock_data.release()
        return shape
    provide_data = property(provide_data_func)

    def provide_label_func(self, data=None):
        self.lock_data.acquire()
        if data is None:
            data = self.data
        shape = [None for _ in range(len(data))]
        self.lock_data.release()
        return shape
    provide_label = property(provide_label_func)

    def provide_data_single_func(self, data=None):
        self.lock_data.acquire()
        if data is None:
            data = self.data
        shape = [(k, v.shape) for k, v in zip(self.data_name, data[0])]
        self.lock_data.release()
        return shape
    provide_data_single = property(provide_data_single_func)

    def provide_label_single_func(self, data=None):
        self.lock_data.acquire()
        shape = None
        self.lock_data.release()
        return shape
    provide_label_single = property(provide_label_single_func)

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur < self.size

    def next(self):
        self.lock.acquire()
        if self.iter_next():
            cur_from = self.cur
            self.cur += self.batch_size
            self.lock.release()
            data, im_info = self.get_batch(cur_from)
            return im_info, mx.io.DataBatch(data=data, label=self.label,
                   pad=self.getpad(cur_from), index=self.getindex(cur_from),
                   provide_data=self.provide_data_func(data), provide_label=self.provide_label_func(data))
        else:
            self.lock.release()
            raise StopIteration

    def getindex(self, cur_from=None):
        if cur_from is None:
            cur_from = self.cur
        return cur_from / self.batch_size

    def getpad(self, cur_from=None):
        if cur_from is None:
            cur_from = self.cur
        if cur_from + self.batch_size > self.size:
            return cur_from + self.batch_size - self.size
        else:
            return 0

    def get_batch(self, cur_from=None):
        if cur_from is None:
            cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        if self.has_rpn:
            data, label, im_info = get_rpn_testbatch(roidb, self.cfg)
        else:
            data, label, im_info = get_rcnn_testbatch(roidb, self.cfg)
        data = [[mx.nd.array(idata[name]) for name in self.data_name] for idata in data]
        self.lock_data.acquire()
        self.data = data
        self.im_info = im_info
        self.lock_data.release()

        return data, im_info

    def get_batch_individual(self, cur_from=None):
        if cur_from is None:
            cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        if self.has_rpn:
            data, label, im_info = get_rpn_testbatch(roidb, self.cfg)
        else:
            data, label, im_info = get_rcnn_testbatch(roidb, self.cfg)

        self.lock_data.acquire()
        self.data = [mx.nd.array(data[name]) for name in self.data_name]
        self.im_info = im_info
        self.lock_data.release()

        return data, im_info


class ROIIter(mx.io.DataIter):
    def __init__(self, roidb, config, batch_size=2, shuffle=False, ctx=None, work_load_list=None, aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: ROIIter
        """
        super(ROIIter, self).__init__()

        # save parameters as properties
        self.roidb = roidb
        self.cfg = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names (only for training)
        if self.cfg.network.ROIDispatch:
            self.data_name = ['data', 'rois_0', 'rois_1', 'rois_2', 'rois_3']
        else:
            self.data_name = ['data', 'rois']

        if self.cfg.TRAIN.LEARN_NMS:
            self.data_name.append('im_info')
            self.data_name.append('gt_boxes')

        if self.cfg.network.USE_NONGT_INDEX:
            self.label_name = ['label', 'nongt_index', 'bbox_target', 'bbox_weight']
        else:
            self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        self.lock = threading.Lock()
        self.lock_data = threading.RLock()

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_individual()

    def provide_data_func(self, data=None):
        self.lock_data.acquire()
        if data is None:
            data = self.data
        shape = [[(k, v.shape) for k, v in zip(self.data_name, data[i])] for i in xrange(len(data))]
        self.lock_data.release()
        return shape
    provide_data = property(provide_data_func)

    def provide_label_func(self, label=None):
        self.lock_data.acquire()
        if label is None:
            label = self.label
        shape = [[(k, v.shape) for k, v in zip(self.label_name, label[i])] for i in xrange(len(label))]
        self.lock_data.release()
        return shape
    provide_label = property(provide_label_func)

    def provide_data_single_func(self, data=None):
        self.lock_data.acquire()
        if data is None:
            data = self.data
        shape = [(k, v.shape) for k, v in zip(self.data_name, data[0])]
        self.lock_data.release()
        return shape
    provide_data_single = property(provide_data_single_func)

    def provide_label_single_func(self, label=None):
        self.lock_data.acquire()
        if label is None:
            label = self.label
        shape = [(k, v.shape) for k, v in zip(self.label_name, label[0])]
        self.lock_data.release()
        return shape
    provide_label_single = property(provide_label_single_func)

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        self.lock.acquire()
        if self.iter_next():
            cur_from = self.cur
            self.cur += self.batch_size
            self.lock.release()
            data, label = self.get_batch_individual(cur_from)
            return mx.io.DataBatch(data=data, label=label,
               pad=self.getpad(cur_from), index=self.getindex(cur_from),
               provide_data=self.provide_data_func(data), provide_label=self.provide_label_func(label))
        else:
            self.lock.release()
            raise StopIteration

    def getindex(self, cur_from=None):
        if cur_from is None:
            cur_from = self.cur
        return cur_from / self.batch_size

    def getpad(self, cur_from=None):
        if cur_from is None:
            cur_from = self.cur
        if cur_from + self.batch_size > self.size:
            return cur_from + self.batch_size - self.size
        else:
            return 0

    def get_batch(self, cur_from=None):
        # slice roidb
        if cur_from is None:
            cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slices
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        # get each device
        data_list = []
        label_list = []
        for islice in slices:
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            data, label = get_rcnn_batch(iroidb, self.cfg)
            data_list.append(data)
            label_list.append(label)

        all_data = dict()
        for key in data_list[0].keys():
            all_data[key] = tensor_vstack([batch[key] for batch in data_list])

        all_label = dict()
        for key in label_list[0].keys():
            all_label[key] = tensor_vstack([batch[key] for batch in label_list])

        data = [mx.nd.array(all_data[name]) for name in self.data_name]
        label = [mx.nd.array(all_label[name]) for name in self.label_name]
        
        self.lock_data.acquire()
        self.data = data
        self.label = label
        self.lock_data.release()

        return data, label

    def get_batch_individual(self, cur_from=None):
        if cur_from is None:
            cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]

        # decide multi device slices
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)

        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(self.parfetch(iroidb))

        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

        self.lock_data.acquire()
        self.data = data
        self.label = label
        self.lock_data.release()

        return data, label

    #def parfetch(self, iroidb):
    #    data, label = get_rcnn_batch(iroidb, self.cfg)
    #    return {'data': data, 'label': label}

    def parfetch(self, iroidb):
        # get testing data for multigpu
        data, label = get_rcnn_batch(iroidb, self.cfg)

        # add gt_boxes to data for learn nms
        if 'gt_boxes' in data:
            data['gt_boxes'] = data['gt_boxes'][np.newaxis, :, :]

        return {'data': data, 'label': label}


class AnchorLoader(mx.io.DataIter):

    def __init__(self, feat_sym, roidb, cfg, batch_size=1, shuffle=False, ctx=None, work_load_list=None,
                 feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2), allowed_border=0,
                 aspect_grouping=False):
        """
        This Iter will provide roi data to Fast R-CNN network
        :param feat_sym: to infer shape of assign_output
        :param roidb: must be preprocessed
        :param batch_size: must divide BATCH_SIZE(128)
        :param shuffle: bool
        :param ctx: list of contexts
        :param work_load_list: list of work load
        :param aspect_grouping: group images with similar aspects
        :return: AnchorLoader
        """
        super(AnchorLoader, self).__init__()

        # save parameters as properties
        self.feat_sym = feat_sym
        self.roidb = roidb
        self.cfg = cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = [mx.cpu()]
        self.work_load_list = work_load_list
        self.feat_stride = feat_stride
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.allowed_border = allowed_border
        self.aspect_grouping = aspect_grouping

        # infer properties from roidb
        self.size = len(roidb)
        self.index = np.arange(self.size)

        # decide data and label names
        if config.TRAIN.END2END:
            self.data_name = ['data', 'im_info', 'gt_boxes']
        else:
            self.data_name = ['data']
        self.label_name = ['label', 'bbox_target', 'bbox_weight']

        # status variable for synchronization between get_data and get_label
        self.cur = 0
        self.batch = None
        self.data = None
        self.label = None

        self.lock = threading.Lock()
        self.lock_data = threading.RLock()

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self.get_batch_individual()

    def provide_data_func(self, data=None):
        self.lock_data.acquire()
        if data is None:
            data = self.data
        shape = [[(k, v.shape) for k, v in zip(self.data_name, data[i])] for i in xrange(len(data))]
        self.lock_data.release()
        return shape
    provide_data = property(provide_data_func)

    def provide_label_func(self, label=None):
        self.lock_data.acquire()
        if label is None:
            label = self.label
        shape = [[(k, v.shape) for k, v in zip(self.label_name, label[i])] for i in xrange(len(label))]
        self.lock_data.release()
        return shape
    provide_label = property(provide_label_func)

    def provide_data_single_func(self, data=None):
        self.lock_data.acquire()
        if data is None:
            data = self.data
        shape = [(k, v.shape) for k, v in zip(self.data_name, data[0])]
        self.lock_data.release()
        return shape
    provide_data_single = property(provide_data_single_func)

    def provide_label_single_func(self, label=None):
        self.lock_data.acquire()
        if label is None:
            label = self.label
        shape = [(k, v.shape) for k, v in zip(self.label_name, label[0])]
        self.lock_data.release()
        return shape
    provide_label_single = property(provide_label_single_func)

    def reset(self):
        self.cur = 0
        if self.shuffle:
            if self.aspect_grouping:
                widths = np.array([r['width'] for r in self.roidb])
                heights = np.array([r['height'] for r in self.roidb])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((np.random.permutation(horz_inds), np.random.permutation(vert_inds)))
                extra = inds.shape[0] % self.batch_size
                inds_ = np.reshape(inds[:-extra], (-1, self.batch_size))
                row_perm = np.random.permutation(np.arange(inds_.shape[0]))
                inds[:-extra] = np.reshape(inds_[row_perm, :], (-1,))
                self.index = inds
            else:
                np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def next(self):
        self.lock.acquire()
        if self.iter_next():
            cur_from = self.cur
            self.cur += self.batch_size
            self.lock.release()
            data, label = self.get_batch_individual(cur_from)
            return mx.io.DataBatch(data=data, label=label,
               pad=self.getpad(cur_from), index=self.getindex(cur_from),
               provide_data=self.provide_data_func(data), provide_label=self.provide_label_func(label))
        else:
            self.lock.release()
            raise StopIteration

    def getindex(self, cur_from=None):
        if cur_from is None:
            cur_from = self.cur
        return cur_from / self.batch_size

    def getpad(self, cur_from=None):
        if cur_from is None:
            cur_from = self.cur
        if cur_from + self.batch_size > self.size:
            return cur_from + self.batch_size - self.size
        else:
            return 0

    def infer_shape(self, max_data_shape=None, max_label_shape=None):
        """ Return maximum data and label shape for single gpu """
        if max_data_shape is None:
            max_data_shape = []
        if max_label_shape is None:
            max_label_shape = []
        max_shapes = dict(max_data_shape + max_label_shape)
        input_batch_size = max_shapes['data'][0]
        im_info = [[max_shapes['data'][2], max_shapes['data'][3], 1.0]]
        _, feat_shape, _ = self.feat_sym.infer_shape(**max_shapes)
        label = assign_anchor(feat_shape[0], np.zeros((0, 5)), im_info, self.cfg,
                              self.feat_stride, self.anchor_scales, self.anchor_ratios, self.allowed_border)
        label = [label[k] for k in self.label_name]
        label_shape = [(k, tuple([input_batch_size] + list(v.shape[1:]))) for k, v in zip(self.label_name, label)]
        return max_data_shape, label_shape

    def get_batch_individual(self, cur_from=None):
        if cur_from is None:
            cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        roidb = [self.roidb[self.index[i]] for i in range(cur_from, cur_to)]
        # decide multi device slice
        work_load_list = self.work_load_list
        ctx = self.ctx
        if work_load_list is None:
            work_load_list = [1] * len(ctx)
        assert isinstance(work_load_list, list) and len(work_load_list) == len(ctx), \
            "Invalid settings for work load. "
        slices = _split_input_slice(self.batch_size, work_load_list)
        rst = []
        for idx, islice in enumerate(slices):
            iroidb = [roidb[i] for i in range(islice.start, islice.stop)]
            rst.append(self.parfetch(iroidb))
        all_data = [_['data'] for _ in rst]
        all_label = [_['label'] for _ in rst]
        data = [[mx.nd.array(data[key]) for key in self.data_name] for data in all_data]
        label = [[mx.nd.array(label[key]) for key in self.label_name] for label in all_label]

        self.lock_data.acquire()
        self.data = data
        self.label = label
        self.lock_data.release()

        return data, label

    def parfetch(self, iroidb):
        # get testing data for multigpu
        data, label = get_rpn_batch(iroidb, self.cfg)
        data_shape = {k: v.shape for k, v in data.items()}
        del data_shape['im_info']
        _, feat_shape, _ = self.feat_sym.infer_shape(**data_shape)
        feat_shape = [int(i) for i in feat_shape[0]]

        # add gt_boxes to data for e2e
        data['gt_boxes'] = label['gt_boxes'][np.newaxis, :, :]

        # assign anchor for label
        label = assign_anchor(feat_shape, label['gt_boxes'], data['im_info'], self.cfg,
                              self.feat_stride, self.anchor_scales,
                              self.anchor_ratios, self.allowed_border)
        return {'data': data, 'label': label}


