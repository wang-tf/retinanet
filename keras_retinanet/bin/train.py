#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import print_function

import functools
import os
import sys
import shutil
import warnings

import keras
import keras.preprocessing.image
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.python.client import timeline

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    # import keras_retinanet.bin
    #
    # __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.

import losses
import models
from models.retinanet import retinanet_bbox
from utils.anchors import make_shapes_callback, anchor_targets_bbox
from utils.customized_optimizer import AdamWithLRMult
from utils.keras_version import check_keras_version
#from utils.parse_config import parse_config
from utils.creat_callbacks import create_callbacks


from pascal_voc import PascalVocGenerator
from utils.transform import random_transform_generator


import json


def parse_config(config_path):
    json_data_file = open(config_path)
    configs = json.load(json_data_file)
    json_data_file.close()
    # DatasetConfigs = configs['Dataset']
    # TrainConfigs = configs['Train']
    # DataAugConfigs = configs['Data_Augmentation']
    return configs


def freeze_model(model):
    """ Set all layers in a model to non-trainable.

    The weights for these layers will not be updated during training.

    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    """
    for layer in model.layers:
        layer.trainable = False
    return model


def create_generators(configs):
    """create a input data generator"""
    # create random transform generator for augmenting training data
    if not configs["Data_Augmentation"]['only_x_flip']:
        transform_generator = random_transform_generator(
            min_rotation=configs['Data_Augmentation']['rotation'][0],
            max_rotation=configs['Data_Augmentation']['rotation'][1],
            min_translation=configs['Data_Augmentation']['min_translation'],
            max_translation=configs['Data_Augmentation']['max_translation'],
            min_shear=configs['Data_Augmentation']['shear'][0],
            max_shear=configs['Data_Augmentation']['shear'][1],
            min_scaling=configs['Data_Augmentation']['min_scaling'],
            max_scaling=configs['Data_Augmentation']['max_scaling'],
            flip_x_chance=0.5,
            gray=configs['Data_Augmentation']['gray'],
            inverse_color=configs['Data_Augmentation']['inverse_color'],
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)

    if configs['Dataset']['dataset_type'] == 'pascal':
        train_generator = PascalVocGenerator(
            configs['Dataset']['dataset_path'],
            'trainval',
            classes=configs['Dataset']['classes'],
            transform_generator=transform_generator,
            batch_size=configs['Train']['batch_size'],
            image_min_side=configs['Train']['image_min_side'],
            image_max_side=configs['Train']['image_max_side'],
            sizes=configs['Anchors']['sizes'],
            strides=configs['Anchors']['strides'],
            ratios=configs['Anchors']['ratios'],
            scales=configs['Anchors']['scales'],
        )

        validation_generator = PascalVocGenerator(
            configs['Dataset']['dataset_path'],
            'test',
            classes=configs['Dataset']['classes'],
            batch_size=configs['Train']['batch_size'],
            image_min_side=configs['Train']['image_min_side'],
            image_max_side=configs['Train']['image_max_side'],
            sizes=configs['Anchors']['sizes'],
            strides=configs['Anchors']['strides'],
            ratios=configs['Anchors']['ratios'],
            scales=configs['Anchors']['scales'],
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(configs['Dataset']['dataset_type']))

    return train_generator, validation_generator


def get_session():
    """set tensorflow session"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def model_with_weights(model, weights, skip_mismatch):
    """load model's weight and return model"""
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def main(config_file=None):
    cwd = os.getcwd()

    # parse configuration file
    if config_file is None:
        config_file = sys.argv[-1]
        config_file = os.path.join(cwd, config_file)
    config_file_name = config_file.split('/')[-1]
    configs = parse_config(config_file)

    # save config file
    if configs['Train']['save_configs']:
        # confirm save dir
        config_save_path = 'logs/' + configs['Name']
        config_save_path = os.path.join(cwd, config_save_path)
        if not os.path.exists(config_save_path):
            os.mkdir(os.path.join(cwd, config_save_path))
        # copy config file
        config_dst_name = configs['Name'] + '.json'
        config_file_dst = os.path.join(config_save_path, config_dst_name)
        shutil.copy(config_file, config_file_dst)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if configs['Train']['gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs['Train']['gpu']
    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator, validation_generator = create_generators(configs)

    # create the model
    if configs['Train']['load_snapshot'] is not None:
        print('Loading model, this may take a second...')
        model = models.load_model(configs['Train']['load_snapshot'],
                                  backbone=configs['Train']['backbone'])
        training_model = prediction_model = model
    else:
        weights = configs['Train']['weights']
        # default to imagenet if nothing else is specified
        if weights is None and configs['Train']['imagenet_weights']:
            weights = models.download_imagenet(configs['Train']['backbone'])

        print('Creating model, this may take a second...')
        backbone = configs['Train']['backbone']
        num_classes = train_generator.num_classes()
        multi_gpu = configs['Train']['multi_gpu']
        freeze_backbone = configs['Train']['freeze_backbone']

        modifier = freeze_model if freeze_backbone else None

        # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
        # optionally wrap in a parallel model
        if multi_gpu > 1:
            with tf.device('/cpu:0'):
                retinanet = models.retinanet_backbone(configs['Train']['backbone'])(num_classes,backbone=backbone,modifier=modifier)
                model = model_with_weights(retinanet, weights=weights, skip_mismatch=True)
            training_model = multi_gpu_model(model, gpus=multi_gpu)
        else:
            retinanet = models.retinanet_backbone(configs['Train']['backbone'])(num_classes, backbone=backbone, modifier=modifier)
            training_model = model = model_with_weights(retinanet, weights=weights, skip_mismatch=True)

        # make prediction model
        prediction_model = retinanet_bbox(model=model, anchor_param=configs['Anchors'])

        # compile model
        subnet_loss = {'regression': losses.smooth_l1(), 'classification': losses.focal()}

        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()

        if configs['Train']['lr_multiplier_layer']:
            optimizer = AdamWithLRMult(lr=configs['Train']['init_lr'],
                                       lr_multipliers=configs['Train']['lr_multiplier_layer'],
                                       debug_verbose=False,
                                       clipnorm=0.001)
        else:
            optimizer = keras.optimizers.adam(configs['Train']['init_lr'], clipnorm=0.001)

        training_model.compile(loss=subnet_loss, optimizer=optimizer)

        # training_model.compile(loss=subnet_loss, optimizer=keras.optimizers.adam(configs['Train']['init_lr'], clipnorm=0.001))

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in configs['Train']['backbone'] or 'densenet' in configs['Train']['backbone']:
        compute_anchor_targets = functools.partial(anchor_targets_bbox, shapes_callback=make_shapes_callback(model))
        train_generator.compute_anchor_targets = compute_anchor_targets
        if validation_generator is not None:
            validation_generator.compute_anchor_targets = compute_anchor_targets

    # create the callbacks
    callbacks = create_callbacks(model, training_model, prediction_model, validation_generator, configs, )

    # start training
    training_model.fit_generator(train_generator, validation_data=validation_generator, validation_steps=39,
                                 steps_per_epoch=configs['Train']['steps'],
                                 epochs=configs['Train']['epochs'], verbose=1,
                                 callbacks=callbacks, )

    # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    # with open('timeline.ctf.json', 'w') as f:
    #     f.write(trace.generate_chrome_trace_format())


if __name__ == '__main__':
    main()
