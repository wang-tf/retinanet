"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys

sys.path.append("..")

import os
import numpy as np
from six import raise_from
from PIL import Image
import random
import threading
import time
import warnings
import cv2
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import keras

sys.path.append("..")

from utils.anchors import anchor_targets_bbox, bbox_transform
from utils.image import TransformParameters
from utils.image import apply_transform

from utils.transform import transform_aabb
from utils.transform import random_transform_generator
from utils.transform import change_transform_origin


def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    # result = change_transform_origin(transform, (0.5 * width, 0.5 * height))
    translation_l = np.array([
        [1, 0, 0.5 * width],
        [0, 1, 0.5 * height],
        [0, 0, 1]
    ])
    translation_r = np.array([
        [1, 0, -0.5 * width],
        [0, 1, -0.5 * height],
        [0, 0, 1]
    ])
    result = np.linalg.multi_dot([translation_l, transform, translation_r])

    return result


class PascalVocGenerator():
    def __init__(
            self,
            data_dir,
            set_name,
            classes,
            image_extension='.jpg',
            skip_truncated=False,
            skip_difficult=False,
            group_method='ratio',
            shuffle_groups=True,
            transform_parameters=None,
            compute_anchor_targets=anchor_targets_bbox,
            transform_generator=None,
            **kwargs
    ):
        self.data_dir = data_dir
        self.set_name = set_name
        self.classes = classes
        self.image_names = [l.strip().split(None, 1)[0] for l in open(os.path.join(data_dir, 'ImageSets', 'Main', set_name + '.txt')).readlines()]
        self.image_extension = image_extension
        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # super(PascalVocGenerator, self).__init__(**kwargs)
        self.transform_generator = transform_generator
        self.batch_size = int(kwargs['batch_size'])
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.image_min_side = kwargs['image_min_side']
        self.image_max_side = kwargs['image_max_side']
        self.sizes = kwargs['sizes']
        self.strides = kwargs['strides']
        self.ratios = kwargs['ratios']
        self.scales = kwargs['scales']

        self.transform_parameters = transform_parameters or TransformParameters()
        self.compute_anchor_targets = compute_anchor_targets

        self.group_index = 0
        self.lock = threading.Lock()

        self.group_images()

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.image_names)))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return len(self.classes)

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        image = Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        # return read_image_bgr(path)
        image = np.asarray(Image.open(path).convert('RGB'))
        return image[:, :, ::-1].copy()

    def __parse_annotation(self, element):
        truncated = _findNode(element, 'truncated', parse=int)
        difficult = _findNode(element, 'difficult', parse=int)

        class_name = _findNode(element, 'name').text
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))

        box = np.zeros((1, 5))
        box[0, 4] = self.name_to_label(class_name)

        bndbox = _findNode(element, 'bndbox')  # start from one
        box[0, 0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float) - 1
        box[0, 1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float) - 1
        box[0, 2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float) - 1
        box[0, 3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float) - 1

        return truncated, difficult, box

    def __parse_annotations(self, xml_root):
        size_node = _findNode(xml_root, 'size')
        width = _findNode(size_node, 'width', 'size.width', parse=float)
        height = _findNode(size_node, 'height', 'size.height', parse=float)

        boxes = np.zeros((0, 5))
        for i, element in enumerate(xml_root.iter('object')):
            try:
                truncated, difficult, box = self.__parse_annotation(element)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)
                continue
            if truncated and self.skip_truncated:
                continue
            if difficult and self.skip_difficult:
                continue
            boxes = np.append(boxes, box, axis=0)

        return boxes

    def load_annotations(self, image_index):
        filename = self.image_names[image_index] + '.xml'
        try:
            tree = ET.parse(os.path.join(self.data_dir, 'Annotations', filename))
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)

    def load_annotations_group(self, group):
        return [self.load_annotations(image_index) for image_index in group]

    def filter_annotations(self, image_group, annotations_group, group):
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            assert (isinstance(annotations, np.ndarray)), '\'load_annotations\' should return a list of numpy arrays, received: {}'.format(type(annotations))

            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations[:, 2] <= annotations[:, 0]) |  # xmax < xmin
                (annotations[:, 3] <= annotations[:, 1]) |  # ymax < ymin
                (annotations[:, 0] < 0) |  # xmin < 0
                (annotations[:, 1] < 0) |  # ymin < 0
                (annotations[:, 2] > image.shape[1]) |  # xmax > cols
                (annotations[:, 3] > image.shape[0])    # ymax > rows
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    [annotations[invalid_index, :] for invalid_index in invalid_indices]
                ))
                annotations_group[index] = np.delete(annotations, invalid_indices, axis=0)

        return image_group, annotations_group

    def load_image_group(self, group):
        return [self.load_image(image_index) for image_index in group]

    def random_transform_group_entry(self, image, annotations):
        # randomly transform both image and annotations
        if self.transform_generator:
            _, _, transform_mat = next(self.transform_generator)
            transform = adjust_transform_for_image(transform_mat, image, self.transform_parameters.relative_translation)

            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations = annotations.copy()
            for index in range(annotations.shape[0]):
                annotations[index, :4] = transform_aabb(transform, annotations[index, :4])

        return image, annotations

    def resize_image(self, img):
        # return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)
        (rows, cols, _) = img.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = self.image_min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)
        if largest_side * scale > self.image_max_side:
            scale = self.image_max_side / largest_side

        # resize the image with the computed scale
        img = cv2.resize(img, None, fx=scale, fy=scale)

        return img, scale

    def preprocess_image(self, image):
        # mostly identical to "https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py"
        # except for converting RGB -> BGR since we assume BGR already
        # channels_mean = [114.86156857307878, 121.22668756183155, 125.8848528128925]
        channels_mean = [0, 0, 0]
        image = image.astype('float32')
        image[..., 0] -= channels_mean[0]
        image[..., 1] -= channels_mean[1]
        image[..., 2] -= channels_mean[2]
        return image

    def _show_annotated_image_debug(self, image, annotations):
        draw = image.copy()
        for annotation in annotations:
            annotation = [int(a) for a in annotation]
            if annotation[-1] == 0:
                cv2.rectangle(draw, (annotation[0], annotation[1]), (annotation[2], annotation[3]), (0, 0, 255), 2)
            else:
                cv2.rectangle(draw, (annotation[0], annotation[1]), (annotation[2], annotation[3]), (0, 255, 0), 2)
        cv2.imshow("a", draw)
        cv2.waitKey(0)

    def preprocess_group_entry(self, image, annotations):
        # preprocess the image
        image = self.preprocess_image(image)

        # self._show_annotated_image_debug(image, annotations)

        # randomly transform image and annotations
        image, annotations = self.random_transform_group_entry(image, annotations)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations[:, :4] *= image_scale

        # self._show_annotated_image_debug(image, annotations)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # preprocess a single group entry
            image, annotations = self.preprocess_group_entry(image, annotations)

            # copy processed data back to group
            image_group[index] = image
            annotations_group[index] = annotations

        return image_group, annotations_group

    def compute_inputs(self, image_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((self.batch_size,) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def compute_targets(self, image_group, annotations_group):
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # compute labels and regression targets
        labels_group = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # compute regression targets
            labels_group[index], annotations, anchors = self.compute_anchor_targets(
                max_shape,
                annotations,
                self.num_classes(),
                mask_shape=image.shape,
                sizes=self.sizes,
                strides=self.strides,
                ratios=self.ratios,
                scales=self.scales,
            )
            regression_group[index] = bbox_transform(anchors, annotations)

            # append anchor states to regression targets (necessary for filtering 'ignore', 'positive' and 'negative' anchors)
            anchor_states = np.max(labels_group[index], axis=1, keepdims=True)
            regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

        labels_batch = np.zeros((self.batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())

        # copy all labels and regression values to the batch blob
        for index, (labels, regression) in enumerate(zip(labels_group, regression_group)):
            labels_batch[index, ...] = labels
            regression_batch[index, ...] = regression

        return [regression_batch, labels_batch]

    def compute_input_output(self, group):
        # load images and annotations
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def __next__(self):
        return self.next()

    def next(self):
        # advance the group index
        with self.lock:
            if self.group_index == 0 and self.shuffle_groups:
                # shuffle groups at start of epoch
                random.shuffle(self.groups)
            group = self.groups[self.group_index]
            self.group_index = (self.group_index + 1) % len(self.groups)

        return self.compute_input_output(group)