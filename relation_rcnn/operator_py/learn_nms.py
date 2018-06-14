# --------------------------------------------------------
# Relation Networks for Object Detection
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiayuan Gu, Dazhi Cheng
# --------------------------------------------------------

"""
learn nms operator takes class score, bbox prediction, rois and fc_2_all_relu feature as input,
and outputs final nms score for each roi
this operator is only used in testing stage to speed up nms,
and could be used also in training after gradient calculation for inputs and params is implemented.
"""

import mxnet as mx
from mxnet import nd
import numpy as np
import math
import time

def extract_pairwise_multi_position_embedding_nd(position_mat, feat_dim, wave_length=1000):
    """ Extract multi-class position embedding

    Args:
        position_mat: [num_fg_classes, num_rois, num_rois, 4]
        feat_dim: dimension of embedding feature
        wave_length:

    Returns:
        embedding: [num_fg_classes, num_rois, num_rois, feat_dim]
    """
    feat_range = nd.arange(0, feat_dim / 8)
    dim_mat = nd.broadcast_power(lhs=nd.full((1,), wave_length),
                                     rhs=(8. / feat_dim) * feat_range)
    dim_mat = nd.Reshape(dim_mat, shape=(1, 1, 1, 1, -1))
    position_mat = nd.expand_dims(100.0 * position_mat, axis=4)
    div_mat = nd.broadcast_div(lhs=position_mat, rhs=dim_mat)
    sin_mat = nd.sin(data=div_mat)
    cos_mat = nd.cos(data=div_mat)
    # embedding, [num_fg_classes, num_rois, num_rois, 4, feat_dim/4]
    embedding = nd.concat(sin_mat, cos_mat, dim=4)
    embedding = nd.Reshape(embedding, shape=(0, 0, 0, feat_dim))
    return embedding

def nms_attention_nd(roi_feat, position_mat, nms_pair_pos_fc1_1_weight, nms_pair_pos_fc1_1_bias,
        nms_query_1_weight, nms_query_1_bias, nms_key_1_weight, nms_key_1_bias,
        nms_linear_out_1_weight, nms_linear_out_1_bias, num_rois, dim=(1024, 1024, 1024), 
        fc_dim=(64, 16), feat_dim=1024, group=16, index=1):
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
    roi_feat = nd.transpose(roi_feat, axes=(1, 0, 2))
    # roi_feat_reshape, [num_fg_classes*num_rois, feat_dim]
    roi_feat_reshape = nd.Reshape(roi_feat, shape=(-3, -2))
    # position_embedding, [num_fg_classes, num_rois, num_rois, fc_dim[0]]
    position_embedding = extract_pairwise_multi_position_embedding_nd(position_mat, fc_dim[0])
    # [num_fg_classes * num_rois * num_rois, fc_dim[0]]
    position_embedding_reshape = nd.Reshape(position_embedding, shape=(-1, fc_dim[0]))
    # position_feat_1, [num_fg_classes * num_rois * num_rois, fc_dim[1]]
    position_feat_1 = nd.FullyConnected(name='nms_pair_pos_fc1_' + str(index),
        data=position_embedding_reshape, weight=nms_pair_pos_fc1_1_weight, 
        bias=nms_pair_pos_fc1_1_bias, num_hidden=fc_dim[1])
    # position_feat_1, [num_fg_classes, num_rois, num_rois, fc_dim[1]]
    position_feat_1 = nd.Reshape(position_feat_1, shape=(-1, num_rois, num_rois, fc_dim[1]))
    aff_weight = nd.Activation(data=position_feat_1, act_type='relu')
    # aff_weight, [num_fg_classes, fc_dim[1], num_rois, num_rois]
    aff_weight = nd.transpose(aff_weight, axes=(0, 3, 1, 2))

    ####################### multi head in batch###########################
    assert dim[0] == dim[1], 'Matrix multi requires the same dims!'
    # q_data, [num_fg_classes * num_rois, dim[0]]
    q_data = nd.FullyConnected(name='nms_query_' + str(index), data=roi_feat_reshape, 
        weight=nms_query_1_weight, bias=nms_query_1_bias, num_hidden=dim[0])
    # q_data, [num_fg_classes, num_rois, group, dim_group[0]]
    q_data_batch = nd.Reshape(q_data, shape=(-1, num_rois, group, dim_group[0]))
    q_data_batch = nd.transpose(q_data_batch, axes=(0, 2, 1, 3))
    # q_data_batch, [num_fg_classes * group, num_rois, dim_group[0]]
    q_data_batch = nd.Reshape(q_data_batch, shape=(-3, -2))
    k_data = nd.FullyConnected(name='nms_key_' + str(index), data=roi_feat_reshape, 
        weight=nms_key_1_weight, bias=nms_key_1_bias, num_hidden=dim[1])
    # k_data, [num_fg_classes, num_rois, group, dim_group[1]]
    k_data_batch = nd.Reshape(k_data, shape=(-1, num_rois, group, dim_group[1]))
    k_data_batch = nd.transpose(k_data_batch, axes=(0, 2, 1, 3))
    # k_data_batch, [num_fg_classes * group, num_rois, dim_group[1]]
    k_data_batch = nd.Reshape(k_data_batch, shape=(-3, -2))
    v_data = roi_feat
    aff = nd.batch_dot(lhs=q_data_batch, rhs=k_data_batch, transpose_a=False, transpose_b=True)
    # aff_scale, [num_fg_classes * group, num_rois, num_rois]
    aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff

    assert fc_dim[1] == group, 'Check the dimensions in attention!'
    # [num_fg_classes * fc_dim[1], num_rois, num_rois]
    aff_weight_reshape = nd.Reshape(aff_weight, shape=(-3, -2))
    # weighted_aff, [num_fg_classes * fc_dim[1], num_rois, num_rois]
    weighted_aff= nd.log(nd.maximum(aff_weight_reshape, 1e-6)) + aff_scale
    # aff_softmax, [num_fg_classes * fc_dim[1], num_rois, num_rois]
    aff_softmax = nd.softmax(data=weighted_aff, axis=2, name='nms_softmax_' + str(index))
    aff_softmax_reshape = nd.Reshape(aff_softmax, shape=(-1, fc_dim[1] * num_rois, 0))
    # output_t, [num_fg_classes, fc_dim[1] * num_rois, feat_dim]
    output_t = nd.batch_dot(lhs=aff_softmax_reshape, rhs=v_data)
    # output_t_reshape, [num_fg_classes, fc_dim[1], num_rois, feat_dim]
    output_t_reshape = nd.Reshape(output_t, shape=(-1, fc_dim[1], num_rois, feat_dim))
    # output_t_reshape, [fc_dim[1], feat_dim, num_rois, num_fg_classes]
    output_t_reshape = nd.transpose(output_t_reshape, axes=(1, 3, 2, 0))
    # output_t_reshape, [1, fc_dim[1] * feat_dim, num_rois, num_fg_classes]
    output_t_reshape = nd.Reshape(output_t_reshape, shape=(1, fc_dim[1] * feat_dim, num_rois, -1))
    linear_out = nd.Convolution(name='nms_linear_out_' + str(index), data=output_t_reshape,
        weight=nms_linear_out_1_weight, bias=nms_linear_out_1_bias, kernel=(1, 1), 
        num_filter=dim[2], num_group=fc_dim[1])
    # [dim[2], num_rois, num_fg_classes]
    linear_out_reshape = nd.Reshape(linear_out, shape=(dim[2], num_rois, -1))
    # [num_rois, num_fg_classes, dim[2]]
    output = nd.transpose(linear_out_reshape, axes=(1, 2, 0))
    return output

def extract_rank_embedding_nd(rank_dim, feat_dim, wave_length=1000):
    rank_range = nd.arange(0, rank_dim)
    feat_range = nd.arange(0, feat_dim / 2)
    dim_mat = nd.broadcast_power(lhs=nd.full((1,), wave_length),
                                     rhs=(2. / feat_dim) * feat_range)
    dim_mat = nd.Reshape(dim_mat, shape=(1, -1))
    rank_mat = nd.expand_dims(rank_range, axis=1)
    div_mat = nd.broadcast_div(lhs=rank_mat, rhs=dim_mat)
    sin_mat = nd.sin(data=div_mat)
    cos_mat = nd.cos(data=div_mat)
    embedding = nd.concat(sin_mat, cos_mat, dim=1)
    return embedding

def extract_multi_position_matrix_nd(bbox):
    bbox = nd.transpose(bbox, axes=(1, 0, 2))
    xmin, ymin, xmax, ymax = nd.split(data=bbox, num_outputs=4, axis=2)
    # [num_fg_classes, num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [num_fg_classes, num_boxes, num_boxes]
    delta_x = nd.broadcast_minus(lhs=center_x, 
        rhs=nd.transpose(center_x, axes=(0, 2, 1)))
    delta_x = nd.broadcast_div(delta_x, bbox_width)
    delta_x = nd.log(nd.maximum(nd.abs(delta_x), 1e-3))

    delta_y = nd.broadcast_minus(lhs=center_y,
        rhs=nd.transpose(center_y, axes=(0, 2, 1)))
    delta_y = nd.broadcast_div(delta_y, bbox_height)
    delta_y = nd.log(nd.maximum(nd.abs(delta_y), 1e-3))

    delta_width = nd.broadcast_div(lhs=bbox_width, 
        rhs=nd.transpose(bbox_width, axes=(0, 2, 1)))
    delta_width = nd.log(delta_width)

    delta_height = nd.broadcast_div(lhs=bbox_height,
        rhs=nd.transpose(bbox_height, axes=(0, 2, 1)))
    delta_height = nd.log(delta_height)
    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        concat_list[idx] = nd.expand_dims(sym, axis=3)
    position_matrix = nd.concat(*concat_list, dim=3)
    return position_matrix


def refine_bbox_nd(bbox, bbox_delta, im_info=None, means=None, stds=None):

    xmin, ymin, xmax, ymax = nd.split(data=bbox, num_outputs=4, axis=1)
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)

    bbox_delta_reshape = nd.Reshape(data=bbox_delta, shape=(0, -1, 4))
    dx, dy, dw, dh = nd.split(data=bbox_delta_reshape, 
        num_outputs=4, axis=2, squeeze_axis=1)
    if (means is not None) and (stds is not None):
        dx = dx * stds[0] + means[0]
        dy = dy * stds[1] + means[1]
        dw = dw * stds[2] + means[2]
        dh = dh * stds[3] + means[3]

    refine_center_x = nd.broadcast_add(lhs=center_x,
        rhs=nd.broadcast_mul(lhs=bbox_width, rhs=dx))
    refine_center_y = nd.broadcast_add(lhs=center_y,
        rhs=nd.broadcast_mul(lhs=bbox_height, rhs=dy))
    refined_width = nd.broadcast_mul(lhs=bbox_width, rhs=nd.exp(dw))
    refined_height = nd.broadcast_mul(lhs=bbox_height, rhs=nd.exp(dh))
    w_offset = 0.5 * (refined_width - 1.)
    h_offset = 0.5 * (refined_height - 1.)
    refined_xmin = nd.expand_dims(refine_center_x - w_offset, axis=1)
    refined_ymin = nd.expand_dims(refine_center_y - h_offset, axis=1)
    refined_xmax = nd.expand_dims(refine_center_x + w_offset, axis=1)
    refined_ymax = nd.expand_dims(refine_center_y + h_offset, axis=1)

    refined_bbox = nd.concat(refined_xmin, refined_ymin, refined_xmax, refined_ymax, dim=1)
    if im_info is not None:
        # assume im_info [[height, width, scale]] with shape (1,3)
        im_hw = nd.slice_axis(im_info, axis=1, begin=0, end=2)
        im_wh = nd.reverse(im_hw, axis=1)
        im_wh = im_wh - 1.
        im_wh = nd.tile(data=im_wh, reps=(1, 2))
        im_wh = nd.Reshape(im_wh, shape=(1, 4, 1))
        refined_bbox = nd.broadcast_minimum(lhs=refined_bbox, rhs=im_wh)
        refined_bbox = nd.broadcast_maximum(lhs=refined_bbox,
            rhs=nd.zeros_like(refined_bbox))
    # print refined_bbox.debug_str()
    return refined_bbox

class LearnNmsOperator(mx.operator.CustomOp):
    def __init__(self, num_fg_classes, bbox_means, bbox_stds, first_n, class_agnostic, num_thresh, class_thresh, 
            nongt_dim=None, has_non_gt_index=False, nms_attention_feat_dim=128, 
            nms_attention_group=16, nms_attention_fc_dim=(64, 16), nms_attention_dim=(1024, 1024, 128)):
        super(LearnNmsOperator, self).__init__()
        self.num_fg_classes = num_fg_classes
        self.nongt_dim = nongt_dim
        self.has_non_gt_index = has_non_gt_index
        self.bbox_means = bbox_means
        self.bbox_stds = bbox_stds
        self.first_n = first_n
        self.class_agnostic = class_agnostic
        self.num_thresh = num_thresh
        self.nms_attention_feat_dim = nms_attention_feat_dim
        self.nms_attention_group = nms_attention_group
        self.nms_attention_fc_dim = nms_attention_fc_dim
        self.nms_attention_dim = nms_attention_dim
        self.class_thresh = class_thresh

    def forward(self, is_train, req, in_data, out_data, aux):
        nms_start_time = time.time()
        #inputs
        cls_score = in_data[0]
        bbox_pred = in_data[1]
        rois = in_data[2]
        im_info = in_data[3]
        fc_all_2_relu = in_data[4]
        nms_rank_weight = in_data[5]
        nms_rank_bias = in_data[6]
        roi_feat_embedding_weight = in_data[7]
        roi_feat_embedding_bias = in_data[8]
        nms_pair_pos_fc1_1_weight = in_data[9]
        nms_pair_pos_fc1_1_bias = in_data[10]
        nms_query_1_weight = in_data[11]
        nms_query_1_bias = in_data[12]
        nms_key_1_weight = in_data[13]
        nms_key_1_bias = in_data[14]
        nms_linear_out_1_weight = in_data[15]
        nms_linear_out_1_bias = in_data[16]
        nms_logit_weight = in_data[17]
        nms_logit_bias = in_data[18]
        if self.has_non_gt_index:
            non_gt_index = in_data[19]
        else:
            non_gt_index = None

        if self.nongt_dim is not None:
            cls_score_nongt = nd.slice_axis(data=cls_score, axis=0, begin=0, end=self.nongt_dim)
            # cls_score_nongt = monitor_wrapper(cls_score_nongt, 'cls_score_nongt')
            bbox_pred_nongt = nd.slice_axis(data=bbox_pred, axis=0, begin=0, end=self.nongt_dim)
        elif non_gt_index is not None:
            cls_score_nongt = nd.take(a=cls_score, indices=non_gt_index)
            bbox_pred_nongt = nd.take(a=bbox_pred, indices=non_gt_index)
        else:
            cls_score_nongt = cls_score
            bbox_pred_nongt = bbox_pred
        bbox_pred_nongt = nd.BlockGrad(bbox_pred_nongt)

        # remove batch idx and gt roi
        sliced_rois = nd.slice_axis(data=rois, axis=1, begin=1, end=None)
        if self.nongt_dim is not None:
            sliced_rois = nd.slice_axis(data=sliced_rois, axis=0, begin=0, end=self.nongt_dim)
        elif non_gt_index is not None:
            sliced_rois = nd.take(a=sliced_rois, indices=non_gt_index)
        # bbox_pred_nobg, [num_rois, 4*(num_reg_classes-1)]
        bbox_pred_nobg = nd.slice_axis(data=bbox_pred_nongt, axis=1, begin=4, end=None)
        # [num_boxes, 4, num_reg_classes-1]
        refined_bbox = refine_bbox_nd(sliced_rois, bbox_pred_nobg, im_info,
            means=self.bbox_means, stds=self.bbox_stds)
        # softmax cls_score to cls_prob, [num_rois, num_classes]
        cls_prob = nd.softmax(data=cls_score_nongt, axis=-1)
        cls_prob_nobg = nd.slice_axis(cls_prob, axis=1, begin=1, end=None)
        sorted_cls_prob_nobg = nd.sort(data=cls_prob_nobg, axis=0, is_ascend=False)
        # sorted_score, [first_n, num_fg_classes]
        sorted_score = nd.slice_axis(sorted_cls_prob_nobg, axis=0,
            begin=0, end=self.first_n, name='sorted_score')
        max_score_per_class = sorted_score.max(axis=0)
        max_score_per_class_numpy = max_score_per_class.asnumpy()

        valid_class_thresh = self.class_thresh
        valid_class_thresh = np.minimum(valid_class_thresh, max_score_per_class_numpy.max())
        valid_class_indices = np.where(max_score_per_class_numpy >= valid_class_thresh)[0]
        invalid_class_indices = np.where(max_score_per_class_numpy < valid_class_thresh)[0]
        num_valid_classes = len(valid_class_indices)
        valid_class_indices_nd = nd.array(valid_class_indices, ctx=sorted_score.context)

        # sort by score
        rank_indices = nd.argsort(data=cls_prob_nobg, axis=0, is_ascend=False)
        # first_rank_indices, [first_n, num_fg_classes]
        first_rank_indices = nd.slice_axis(rank_indices, axis=0, begin=0, end=self.first_n)
        valid_first_rank_indices = first_rank_indices.transpose().take(valid_class_indices_nd).transpose()

        # sorted_bbox, [first_n, num_fg_classes, 4, num_reg_classes-1]
        sorted_bbox = nd.take(a=refined_bbox, indices=first_rank_indices)
        if self.class_agnostic:
            # sorted_bbox, [first_n, num_fg_classes, 4]
            sorted_bbox = nd.Reshape(sorted_bbox, shape=(0, 0, 0), name='sorted_bbox')
        else:
            cls_mask = nd.arange(0, self.num_fg_classes)
            cls_mask = nd.Reshape(cls_mask, shape=(1, -1, 1))
            cls_mask = nd.broadcast_to(cls_mask, shape=(self.first_n, 0, 4))
            # sorted_bbox, [first_n, num_fg_classes, 4]
            sorted_bbox = nd.pick(data=sorted_bbox, name='sorted_bbox', index=cls_mask, axis=3)

        valid_sorted_bbox = sorted_bbox.transpose((1, 0, 2)).take(valid_class_indices_nd).transpose((1, 0, 2))

        # sorted_bbox = monitor_wrapper(sorted_bbox, 'sorted_bbox')
        # nms_rank_embedding, [first_n, 1024]
        nms_rank_embedding = extract_rank_embedding_nd(self.first_n, 1024)
        # nms_rank_feat, [first_n, 1024]
        nms_rank_feat = nd.FullyConnected(name='nms_rank', data=nms_rank_embedding, 
            num_hidden=128, weight=nms_rank_weight, bias=nms_rank_bias)
        # nms_position_matrix, [num_valid_classes, first_n, first_n, 4]
        nms_position_matrix = extract_multi_position_matrix_nd(valid_sorted_bbox)
        # roi_feature_embedding, [num_rois, 1024]
        # fc_all_2_relu = monitor_wrapper(fc_all_2_relu, 'fc_all_2_relu')
        roi_feat_embedding = nd.FullyConnected(name='roi_feat_embedding', 
            data=fc_all_2_relu, num_hidden=128, weight=roi_feat_embedding_weight, 
            bias=roi_feat_embedding_bias)
        # sorted_roi_feat, [first_n, num_valid_classes, 128]
        sorted_roi_feat = nd.take(a=roi_feat_embedding, indices=valid_first_rank_indices)

        # vectorized nms
        # nms_embedding_feat, [first_n, num_valid_classes, 128]
        nms_embedding_feat = nd.broadcast_add(
            lhs=sorted_roi_feat,
            rhs=nd.expand_dims(nms_rank_feat, axis=1))
        # nms_attention_1, [first_n, num_valid_classes, 1024]
        nms_attention_1 = nms_attention_nd(nms_embedding_feat, 
            nms_position_matrix, nms_pair_pos_fc1_1_weight, nms_pair_pos_fc1_1_bias,
            nms_query_1_weight, nms_query_1_bias, nms_key_1_weight, nms_key_1_bias,
            nms_linear_out_1_weight, nms_linear_out_1_bias, num_rois=self.first_n, index=1, 
            group=self.nms_attention_group, dim=self.nms_attention_dim, 
            fc_dim=self.nms_attention_fc_dim, feat_dim=self.nms_attention_feat_dim)
        nms_all_feat_1 = nms_embedding_feat + nms_attention_1
        nms_all_feat_1_relu = nd.Activation(data=nms_all_feat_1, act_type='relu', name='nms_all_feat_1_relu')
        # [first_n * num_valid_classes, 1024]
        nms_all_feat_1_relu_reshape = nd.Reshape(nms_all_feat_1_relu, shape=(-3, -2))
        # logit, [first_n * num_valid_classes, num_thresh]
        nms_conditional_logit = nd.FullyConnected(name='nms_logit',
            data=nms_all_feat_1_relu_reshape, num_hidden=self.num_thresh,
            weight=nms_logit_weight, bias=nms_logit_bias)
        # logit_reshape, [first_n, num_valid_classes, num_thresh]
        nms_conditional_logit_reshape = nd.Reshape(nms_conditional_logit,
            shape=(self.first_n, num_valid_classes, self.num_thresh))
        nms_conditional_score = nd.Activation(data=nms_conditional_logit_reshape,
            act_type='sigmoid', name='nms_conditional_score')
        if num_valid_classes == self.num_fg_classes:
            full_nms_conditional_score = nms_conditional_score
        else:
            full_nms_conditional_score = nd.concat(nms_conditional_score, nd.zeros((self.first_n,
                self.num_fg_classes - num_valid_classes, self.num_thresh), 
                ctx=nms_conditional_score.context), dim=1)

        all_indexes = np.concatenate((valid_class_indices, invalid_class_indices))
        restore_indexes = np.zeros((self.num_fg_classes))
        restore_indexes[all_indexes] = np.arange(self.num_fg_classes)
        restore_indexes = nd.array(restore_indexes, ctx=nms_conditional_score.context)
        full_nms_conditional_score = full_nms_conditional_score.transpose((1, 0, 2)).take(restore_indexes).transpose((1, 0, 2))
        
        sorted_score_reshape = nd.expand_dims(sorted_score, axis=2)
        # sorted_score_reshape = nd.BlockGrad(sorted_score_reshape)
        nms_multi_score = nd.broadcast_mul(lhs=sorted_score_reshape, rhs=full_nms_conditional_score)
        _ = nms_multi_score.mean().asnumpy()

        all_time = time.time() - nms_start_time
        if 'learn_nms_time' not in globals().keys() or 'learn_nms_count' not in globals().keys():
            globals()['learn_nms_time'] = []
            globals()['learn_nms_count'] = 0

        if globals()['learn_nms_count'] >= 1000:
            globals()['learn_nms_time'].pop(0)
            globals()['learn_nms_time'].append(all_time)
        else:
            globals()['learn_nms_time'].append(all_time)

        globals()['learn_nms_count'] += 1
        if globals()['learn_nms_count'] % 250 == 0:
            print("--->> learn nms running average time cost: {}".format(float(sum(globals()['learn_nms_time']))/(1000 if globals()['learn_nms_count'] > 1000 else globals()['learn_nms_count'])))

        self.assign(out_data[0], req[0], nms_multi_score)
        self.assign(out_data[1], req[1], sorted_bbox)
        self.assign(out_data[2], req[2], sorted_score)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for in_grad_single, req_single in zip(in_grad, req):
            self.assign(in_grad_single, req_single, 0)


@mx.operator.register('learn_nms')
class LearnNmsProp(mx.operator.CustomOpProp):
    def __init__(self, num_fg_classes, bbox_means, bbox_stds, first_n, 
            class_agnostic, num_thresh, class_thresh, nongt_dim, has_non_gt_index):
        super(LearnNmsProp, self).__init__(need_top_grad=False)
        self.num_fg_classes = int(num_fg_classes)
        self.nongt_dim = int(nongt_dim) if nongt_dim != 'None' else None
        self.class_thresh = float(class_thresh)
        # gluon customops use , to separate elements, make sure this doesn't happend
        assert ',' not in bbox_means and ',' not in bbox_stds 
        if bbox_means == 'None' or bbox_stds == 'None':
            self.bbox_means = None
            self.bbox_stds = None
        else:
            self.bbox_means = np.fromstring(bbox_means[1:-1], dtype=float, sep=' ')
            self.bbox_stds = np.fromstring(bbox_stds[1:-1], dtype=float, sep=' ')
        self.first_n = int(first_n)
        self.class_agnostic = class_agnostic == 'True'
        self.num_thresh = int(num_thresh)
        self.has_non_gt_index = has_non_gt_index == 'True'

    def list_arguments(self):
        if self.has_non_gt_index:
            return ['cls_score', 'bbox_pred', 'rois', 'im_info', 'fc_all_2_relu', 'nms_rank_weight', 
                'nms_rank_bias', 'roi_feat_embedding_weight', 'roi_feat_embedding_bias', 
                'nms_pair_pos_fc1_1_weight', 'nms_pair_pos_fc1_1_bias', 'nms_query_1_weight', 
                'nms_query_1_bias', 'nms_key_1_weight', 'nms_key_1_bias', 'nms_linear_out_1_weight', 
                'nms_linear_out_1_bias', 'nms_logit_weight', 'nms_logit_bias', 'non_gt_index']
        else:
            return ['cls_score', 'bbox_pred', 'rois', 'im_info', 'fc_all_2_relu', 'nms_rank_weight', 
                'nms_rank_bias', 'roi_feat_embedding_weight', 'roi_feat_embedding_bias', 
                'nms_pair_pos_fc1_1_weight', 'nms_pair_pos_fc1_1_bias', 'nms_query_1_weight', 
                'nms_query_1_bias', 'nms_key_1_weight', 'nms_key_1_bias', 'nms_linear_out_1_weight', 
                'nms_linear_out_1_bias', 'nms_logit_weight', 'nms_logit_bias']

    def list_outputs(self):
        return ['nms_multi_score', 'sorted_bbox', 'sorted_score']

    def infer_shape(self, in_shape):
        nms_multi_score_shape = (self.first_n, self.num_fg_classes, self.num_thresh)
        sorted_bbox_shape = (self.first_n, self.num_fg_classes, 4)
        sorted_score_shape = (self.first_n, self.num_fg_classes)
        return in_shape, [nms_multi_score_shape, sorted_bbox_shape, sorted_score_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return LearnNmsOperator(self.num_fg_classes, self.bbox_means, self.bbox_stds, self.first_n, 
            self.class_agnostic, self.num_thresh, self.class_thresh, self.nongt_dim, self.has_non_gt_index)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []

