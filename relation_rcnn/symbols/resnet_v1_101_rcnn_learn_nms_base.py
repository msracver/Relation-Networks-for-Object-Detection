# --------------------------------------------------------
# Relation Networks for Object Detection
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiayuan Gu, Dazhi Cheng
# --------------------------------------------------------

import cPickle
import mxnet as mx
from utils.symbol import Symbol
from operator_py.proposal import *
from operator_py.proposal_target import *
from operator_py.box_annotator_ohem import *
from resnet_v1_101_rcnn_base import resnet_v1_101_rcnn_base


class resnet_v1_101_rcnn_learn_nms_base(resnet_v1_101_rcnn_base):
    def __init__(self):
        """
        Use __init__ to define parameter network needs
        """
        self.eps = 1e-5
        self.use_global_stats = True
        self.workspace = 512
        self.units = (3, 4, 23, 3)  # use for 101
        self.filter_list = [256, 512, 1024, 2048]

    @staticmethod
    def refine_bbox(bbox, bbox_delta, im_info=None,
                    means=None, stds=None):
        """ Refine bbox according to bbox_delta predictions

        Args:
            bbox: [num_boxes, 4] --> [xmin, ymin, xmax, ymax]
            bbox_delta: [num_boxes, num_reg_classes-1]
            im_info: [..., height, width]
            means: 4-tuple
            stds: 4-tuple

        Returns:
            refined_bbox: [num_boxes, 4, num_reg_classes-1]

        """
        xmin, ymin, xmax, ymax = mx.sym.split(data=bbox,
                                              num_outputs=4, axis=1)
        bbox_width = xmax - xmin + 1.
        bbox_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)

        bbox_delta_reshape = mx.sym.Reshape(data=bbox_delta,
                                            shape=(0, -1, 4))
        dx, dy, dw, dh = mx.sym.split(data=bbox_delta_reshape,
                                      num_outputs=4, axis=2, squeeze_axis=1)
        if (means is not None) and (stds is not None):
            dx = dx * stds[0] + means[0]
            dy = dy * stds[1] + means[1]
            dw = dw * stds[2] + means[2]
            dh = dh * stds[3] + means[3]

        refine_center_x = mx.sym.broadcast_add(lhs=center_x,
                                               rhs=mx.sym.broadcast_mul(lhs=bbox_width, rhs=dx))
        refine_center_y = mx.sym.broadcast_add(lhs=center_y,
                                               rhs=mx.sym.broadcast_mul(lhs=bbox_height, rhs=dy))
        refined_width = mx.sym.broadcast_mul(lhs=bbox_width,
                                             rhs=mx.sym.exp(dw))
        refined_height = mx.sym.broadcast_mul(lhs=bbox_height,
                                              rhs=mx.sym.exp(dh))
        w_offset = 0.5 * (refined_width - 1.)
        h_offset = 0.5 * (refined_height - 1.)
        refined_xmin = mx.sym.expand_dims(refine_center_x - w_offset, axis=1)
        refined_ymin = mx.sym.expand_dims(refine_center_y - h_offset, axis=1)
        refined_xmax = mx.sym.expand_dims(refine_center_x + w_offset, axis=1)
        refined_ymax = mx.sym.expand_dims(refine_center_y + h_offset, axis=1)

        refined_bbox = mx.sym.concat(refined_xmin, refined_ymin, refined_xmax, refined_ymax,
                                     dim=1)
        if im_info is not None:
            # assume im_info [[height, width, scale]] with shape (1,3)
            im_hw = mx.sym.slice_axis(im_info, axis=1, begin=0, end=2)
            im_wh = mx.sym.reverse(im_hw, axis=1)
            im_wh = im_wh - 1.
            im_wh = mx.sym.tile(data=im_wh, reps=(1, 2))
            im_wh = mx.sym.Reshape(im_wh, shape=(1, 4, 1))
            refined_bbox = mx.sym.broadcast_minimum(lhs=refined_bbox,
                                                    rhs=im_wh)
            refined_bbox = mx.sym.broadcast_maximum(lhs=refined_bbox,
                                                    rhs=mx.sym.zeros_like(refined_bbox))
        # print refined_bbox.debug_str()
        return refined_bbox

    @staticmethod
    def compute_overlap(lhs_bbox, rhs_bbox, *args):
        """ Compute overlap of two groups of bbox

        Args:
            lhs_bbox: [num_boxes, num_fg_classes, 4]
            rhs_bbox: [num_boxes, num_fg_classes, 4]
            args: a list of types to compute

        Returns:
            overlap: [num_boxes, num_boxes, num_fg_classes]

        """

        def compute_intersection(bbox1, bbox2):
            bbox1 = mx.sym.transpose(bbox1, axes=(0, 2, 1))
            bbox2 = mx.sym.transpose(bbox2, axes=(2, 0, 1))
            xmin1, ymin1, xmax1, ymax1 = mx.sym.split(
                data=bbox1, num_outputs=4, axis=1)
            xmin2, ymin2, xmax2, ymax2 = mx.sym.split(
                data=bbox2, num_outputs=4, axis=0)
            all_pairs_min_xmax = mx.sym.broadcast_minimum(xmax1, xmax2)
            all_pairs_max_xmin = mx.sym.broadcast_maximum(xmin1, xmin2)
            # [num_boxes, num_boxes, num_fg_classes]
            intersect_widths = mx.sym.maximum(0., all_pairs_min_xmax - all_pairs_max_xmin + 1)
            all_pairs_min_ymax = mx.sym.broadcast_minimum(ymax1, ymax2)
            all_pairs_max_ymin = mx.sym.broadcast_maximum(ymin1, ymin2)
            intersect_heights = mx.sym.maximum(0., all_pairs_min_ymax - all_pairs_max_ymin + 1)
            return intersect_heights * intersect_widths

        def compute_area(bbox):
            xmin, ymin, xmax, ymax = mx.sym.split(
                data=bbox, num_outputs=4, axis=2, squeeze_axis=1)
            bbox_width = xmax - xmin + 1.
            bbox_height = ymax - ymin + 1.
            # [num_boxes, num_fg_classes]
            area = bbox_width * bbox_height
            return area

        intersections = compute_intersection(lhs_bbox, rhs_bbox)
        lhs_area = compute_area(lhs_bbox)
        lhs_area = mx.sym.expand_dims(lhs_area, axis=1)
        rhs_area = compute_area(rhs_bbox)
        rhs_area = mx.sym.expand_dims(rhs_area, axis=0)
        output_list = []
        for overlap_type in args:
            if overlap_type == 'iou':
                unions = mx.sym.broadcast_add(lhs=lhs_area, rhs=rhs_area)
                unions = unions - intersections
                iou = mx.sym.where(condition=(intersections == 0.),
                                   x=mx.sym.zeros_like(intersections),
                                   y=intersections / unions)
                output_list.append(iou)
            elif overlap_type == 'iop':
                iop = mx.sym.where(condition=(intersections == 0.),
                                   x=mx.sym.zeros_like(intersections),
                                   y=mx.sym.broadcast_div(intersections, lhs_area))
                output_list.append(iop)
            elif overlap_type == 'iom':
                min_area = mx.sym.broadcast_minimum(lhs_area, rhs_area)
                iom = mx.sym.where(condition=(intersections == 0.),
                                   x=mx.sym.zeros_like(intersections),
                                   y=mx.sym.broadcast_div(intersections, min_area))
                output_list.append(iom)
            else:
                raise NotImplementedError('Not support computing %s' % overlap_type)
        if len(output_list) > 1:
            return output_list
        elif len(output_list) == 1:
            return output_list[0]
        else:
            raise ValueError('Miss all!')

    @staticmethod
    def extract_rank_embedding(rank_dim, feat_dim, wave_length=1000):
        """ Extract rank embedding

        Args:
            rank_dim: maximum of ranks
            feat_dim: dimension of embedding feature
            wave_length:

        Returns:
            embedding: [rank_dim, feat_dim]
        """
        rank_range = mx.sym.arange(0, rank_dim)
        feat_range = mx.sym.arange(0, feat_dim / 2)
        dim_mat = mx.sym.broadcast_power(lhs=mx.sym.full((1,), wave_length),
                                         rhs=(2. / feat_dim) * feat_range)
        dim_mat = mx.sym.Reshape(dim_mat, shape=(1, -1))
        rank_mat = mx.sym.expand_dims(rank_range, axis=1)
        div_mat = mx.sym.broadcast_div(lhs=rank_mat, rhs=dim_mat)
        sin_mat = mx.sym.sin(data=div_mat)
        cos_mat = mx.sym.cos(data=div_mat)
        embedding = mx.sym.concat(sin_mat, cos_mat, dim=1)
        return embedding

    @staticmethod
    def extract_unary_multi_position_embedding(bbox, feat_dim, wave_length=1000):
        """ Extract multi-class unary position embedding

        Args:
            bbox: [num_boxes, num_classes, 4]
            feat_dim: dimension of embedding feature
            wave_length:

        Returns:
            embedding: [num_boxes, num_classes, feat_dim]
        """
        feat_range = mx.sym.arange(0, feat_dim / 8)
        dim_mat = mx.sym.broadcast_power(lhs=mx.sym.full((1,), wave_length),
                                         rhs=(8. / feat_dim) * feat_range)
        dim_mat = mx.sym.Reshape(dim_mat, shape=(1, 1, -1))
        xmin, ymin, xmax, ymax = mx.sym.split(data=bbox, num_outputs=4, axis=2)
        bbox_width = xmax - xmin + 1
        bbox_height = ymax - ymin + 1
        # remove minus 1 here
        center_x = 0.5 * (xmin + xmax) / 4
        center_y = 0.5 * (ymin + ymax) / 4
        symbol_group = [center_x, center_y, bbox_width, bbox_height]

        embedding_group = list()
        for single_symbol in symbol_group:
            div_mat = mx.sym.broadcast_div(lhs=single_symbol, rhs=dim_mat)
            sin_mat = mx.sym.sin(data=div_mat)
            cos_mat = mx.sym.cos(data=div_mat)
            embedding_group.append(sin_mat)
            embedding_group.append(cos_mat)
        embedding = mx.sym.concat(*embedding_group, dim=2)
        return embedding

    @staticmethod
    def extract_pairwise_multi_position_embedding(position_mat, feat_dim, wave_length=1000):
        """ Extract multi-class position embedding

        Args:
            position_mat: [num_fg_classes, num_rois, num_rois, 4]
            feat_dim: dimension of embedding feature
            wave_length:

        Returns:
            embedding: [num_fg_classes, num_rois, num_rois, feat_dim]
        """
        feat_range = mx.sym.arange(0, feat_dim / 8)
        dim_mat = mx.sym.broadcast_power(lhs=mx.sym.full((1,), wave_length),
                                         rhs=(8. / feat_dim) * feat_range)
        dim_mat = mx.sym.Reshape(dim_mat, shape=(1, 1, 1, 1, -1))
        position_mat = mx.sym.expand_dims(100.0 * position_mat, axis=4)
        div_mat = mx.sym.broadcast_div(lhs=position_mat, rhs=dim_mat)
        sin_mat = mx.sym.sin(data=div_mat)
        cos_mat = mx.sym.cos(data=div_mat)
        # embedding, [num_fg_classes, num_rois, num_rois, 4, feat_dim/4]
        embedding = mx.sym.concat(sin_mat, cos_mat, dim=4)
        embedding = mx.sym.Reshape(embedding, shape=(0, 0, 0, feat_dim))
        return embedding

    @staticmethod
    def extract_multi_position_matrix(bbox):
        """ Extract multi-class position matrix

        Args:
            bbox: [num_boxes, num_fg_classes, 4]

        Returns:
            position_matrix: [num_fg_classes, num_boxes, num_boxes, 4]
        """
        print 'base extract_position_matrix'
        bbox = mx.sym.transpose(bbox, axes=(1, 0, 2))
        xmin, ymin, xmax, ymax = mx.sym.split(data=bbox,
                                              num_outputs=4, axis=2)
        # [num_fg_classes, num_boxes, 1]
        bbox_width = xmax - xmin + 1.
        bbox_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        # [num_fg_classes, num_boxes, num_boxes]
        delta_x = mx.sym.broadcast_minus(lhs=center_x,
                                         rhs=mx.sym.transpose(center_x, axes=(0, 2, 1)))
        delta_x = mx.sym.broadcast_div(delta_x, bbox_width)
        delta_x = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_x), 1e-3))

        delta_y = mx.sym.broadcast_minus(lhs=center_y,
                                         rhs=mx.sym.transpose(center_y, axes=(0, 2, 1)))
        delta_y = mx.sym.broadcast_div(delta_y, bbox_height)
        delta_y = mx.sym.log(mx.sym.maximum(mx.sym.abs(delta_y), 1e-3))

        delta_width = mx.sym.broadcast_div(lhs=bbox_width,
                                           rhs=mx.sym.transpose(bbox_width, axes=(0, 2, 1)))
        delta_width = mx.sym.log(delta_width)

        delta_height = mx.sym.broadcast_div(lhs=bbox_height,
                                            rhs=mx.sym.transpose(bbox_height, axes=(0, 2, 1)))
        delta_height = mx.sym.log(delta_height)
        concat_list = [delta_x, delta_y, delta_width, delta_height]
        for idx, sym in enumerate(concat_list):
            concat_list[idx] = mx.sym.expand_dims(sym, axis=3)
        position_matrix = mx.sym.concat(*concat_list, dim=3)
        return position_matrix

