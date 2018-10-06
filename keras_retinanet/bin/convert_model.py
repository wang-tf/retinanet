#!/usr/bin/env python

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

import argparse
import os
import sys
import keras.models

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
import models
from utils.parse_config import parse_config


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

    parser.add_argument('model_in', help='The model to convert.')
    parser.add_argument('model_out', help='Path to save the converted model to.')
    parser.add_argument('--backbone', help='The backbone of the model to convert.', default='resnet50')
    parser.add_argument('--no-nms', help='Disables non maximum suppression.', dest='nms', action='store_false')
    parser.add_argument('--config_file', help='configuration file during training.', default=None)

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    configs = parse_config(args.config_file)

    # load and convert model
    model = models.load_model(args.model_in, convert=True, backbone=args.backbone, nms=args.nms, anchor_config=configs['Anchors'])

    # save model
    model.save(args.model_out)
    #
    # # load existing (training) model
    # model = keras.models.load_model(args.model_in, custom_objects=models.custom_objects(args.backbone))
    #
    # # wrap it with bbox layers
    # inference_model = models.retinanet.retinanet_bbox(model=model, nms=args.nms)
    # inference_model.save(args.model_out)


if __name__ == '__main__':
    main()
