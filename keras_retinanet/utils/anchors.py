#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np


def shift(shape, stride, anchors):
    """
    Arguments:
        shape: list[int, int], current feature layer's shape.
        stride: int, distance between neigher two anchors.
        anchors: numpy.array, default anchors for any feature map's point.
    Return:
        all_anchors: numpy.array, the whole anchors for input shape.

    >>> shape = [76, 135]
    >>> stride = 8
    >>> anchors = np.array([[          -8.,          -8.,           8.,          8.]
                            [-10.07936817, -10.07936817,  10.07936817, 10.07936817]
                            [-12.69920783, -12.69920783,  12.69920783, 12.69920783]
                            [ -7.15541753,  -8.94427191,   7.15541753,  8.94427191]
                            [ -9.01526096,  -11.2690762,   9.01526096,  11.2690762]
                            [-11.35851679, -14.19814598,  11.35851679, 14.19814598]
                            [ -6.53197265,  -9.79795897,   6.53197265,  9.79795897]
                            [ -8.22976965, -12.34465447,   8.22976965, 12.34465447]
                            [-10.36885977, -15.55328966,  10.36885977, 15.55328966]])
    >>> all_anchors = shift(shape, stride, anchors)
    >>> all_anchors = [[  -4.           -4.           12.           12.        ]
                       [  -6.07936817   -6.07936817   14.07936817   14.07936817]
                       [  -8.69920783   -8.69920783   16.69920783   16.69920783]
                        ...
                       [1069.46802735  594.20204103 1082.53197265  613.79795897]
                       [1067.77023035  591.65534553 1084.22976965  616.34465447]
                       [1065.63114023  588.44671034 1086.36885977  619.55328966]]

    """
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def anchors_for_shape(image_shape, ratios, scales, strides, sizes, pyramid_levels=None, shapes_callback=None,):
    """get all anchors for all pyramid levels"""
    scales = np.array([2 ** i for i in scales])
    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)
    return all_anchors


def anchor_targets_bbox(image_shape, annotations, num_classes, mask_shape=None, negative_overlap=0.4, positive_overlap=0.5, **kwargs):
    anchors = anchors_for_shape(image_shape, **kwargs)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.ones((anchors.shape[0], num_classes)) * -1

    if annotations.shape[0]:
        # obtain indices of gt annotations with the greatest overlap
        overlaps = compute_overlap(anchors, annotations[:, :4])
        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < negative_overlap, :] = 0

        # compute box regression targets
        annotations = annotations[argmax_overlaps_inds]

        # fg label: above threshold IOU
        positive_indices = max_overlaps >= positive_overlap
        labels[positive_indices, :] = 0
        labels[positive_indices, annotations[positive_indices, 4].astype(int)] = 1
    else:
        # no annotations? then everything is background
        labels[:] = 0
        annotations = np.zeros_like(anchors)

    # ignore annotations outside of image
    mask_shape = image_shape if mask_shape is None else mask_shape
    anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
    indices = np.logical_or(anchors_centers[:, 0] >= mask_shape[1], anchors_centers[:, 1] >= mask_shape[0])
    labels[indices, :] = -1

    return labels, annotations, anchors


def layer_shapes(image_shape, model):
    """Compute layer shapes given input image shape and the model.

    :param image_shape:
    :param model:
    :return:
    """
    shape = {model.layers[0].name: (None,) + image_shape,}

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            inputs = [shape[lr.name] for lr in node.inbound_layers]
            if not inputs:
                continue
            shape[layer.name] = layer.compute_output_shape(inputs[0] if len(inputs) == 1 else inputs)

    return shape


def make_shapes_callback(model):
    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels):
    """Guess shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** level - 1) // (2 ** level) for level in pyramid_levels]
    return image_shapes


def generate_anchors(base_size, ratios, scales):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.1, 0.1, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    # clip widths to 1
    gt_widths = np.maximum(gt_widths, 1)
    gt_heights = np.maximum(gt_heights, 1)

    targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
    targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
    targets_dw = np.log(gt_widths / anchor_widths)
    targets_dh = np.log(gt_heights / anchor_heights)

    targets = np.stack((targets_dx, targets_dy, targets_dw, targets_dh))
    targets = targets.T

    targets = (targets - mean) / std

    return targets


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua
