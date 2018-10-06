#!/usr/bin/env python
# coding: utf-8


'''
Input arguments:

num_output: this value has nothing to do with the number of classes, batch_size, etc., 
and it is mostly equal to 1. If the network is a **multi-stream network** 
(forked network with multiple outputs), set the value to the number of outputs.

quantize: if set to True, use the quantize feature of Tensorflow
(https://www.tensorflow.org/performance/quantization) [default: False]

use_theano: Thaeno and Tensorflow implement convolution in different ways.
When using Keras with Theano backend, the order is set to 'channels_first'.
This feature is not fully tested, and doesn't work with quantizization [default: False]
'''


# Parse input arguments

import argparse
import sys
import os

parser = argparse.ArgumentParser(description='set input arguments')
parser.add_argument('-input_fld', help='directory holding the keras weights file [default: .]', action="store", dest='input_fld', type=str, default='./snapshots-v4')
parser.add_argument('-output_fld', help='destination directory to save the tensorflow files [default: .]', action="store", dest='output_fld', type=str, default='')
parser.add_argument('-input_model_file', help="name of the input weight file [default: 'model.h5']", action="store", dest='input_model_file', type=str, default='mobilenet224_1.0_pascal_60_inference.h5')
parser.add_argument('-output_model_file', help="name of the output weight file [default: args.input_model_file + '.pb'] ", action="store", dest='output_model_file', type=str, default='')
parser.add_argument('-output_graphdef_file', help="if graph_def is set to True, the file name of the graph definition [default: model.ascii] ", action="store", dest='output_graphdef_file', type=str, default='model.ascii')
parser.add_argument('-num_outputs', action="store", dest='num_outputs', type=int, default=3)
parser.add_argument('-graph_def', help="if set to True, will write the graph definition as an ascii file [default: False] ", action="store", dest='graph_def', type=bool, default=True)
parser.add_argument('-output_node_prefix', help="the prefix to use for output nodes. [default: output_node] ", action="store", dest='output_node_prefix', type=str, default='output_node')
parser.add_argument('-quantize', action="store", dest='quantize', type=bool, default=False)
parser.add_argument('-theano_backend', action="store", dest='theano_backend', type=bool, default=False)
parser.add_argument('-backbone', type=str, default='mobilenet')
parser.add_argument('-f')
args = parser.parse_args()
parser.print_help()
print('input args: ', args)

if args.theano_backend is True and args.quantize is True:
    raise ValueError("Quantize feature does not work with theano backend.")

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.mobilenet import custom_objects as mobilenet_co
from models.resnet import custom_objects as resnet_co
# initialize

from keras.models import load_model
import tensorflow as tf
from pathlib2 import Path
from keras import backend as K

output_fld =  args.input_fld if args.output_fld == '' else args.output_fld
if args.output_model_file == '':
    args.output_model_file = str(Path(args.input_model_file).name) + '.pb'
Path(output_fld).mkdir(parents=True, exist_ok=True)    
weight_file_path = str(Path(args.input_fld) / args.input_model_file)


# Load keras model and rename output
if 'mobilenet' in args.backbone:
    co = mobilenet_co
elif 'resnet' in args.backbone:
    co = resnet_co
else:
    print('Error backbone: {}'.format(args.backbone))
    # raise

K.set_learning_phase(0)
if args.theano_backend:
    K.set_image_data_format('channels_first')
else:
    K.set_image_data_format('channels_last')

try:
    net_model = load_model(weight_file_path, custom_objects=co)
except ValueError as err:
    print('''Input file specified ({}) only holds the weights, and not the model defenition.
    Save the model using mode.save(filename.h5) which will contain the network architecture
    as well as its weights. 
    If the model is saved using model.save_weights(filename.h5), the model architecture is 
    expected to be saved separately in a json format and loaded prior to loading the weights.
    Check the keras documentation for more details (https://keras.io/getting-started/faq/)'''
          .format(weight_file_path))
    raise err
num_output = args.num_outputs
pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = args.output_node_prefix+str(i)
    pred[i] = tf.identity(net_model.outputs[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)


# [optional] write graph definition in ascii

sess = K.get_session()

if args.graph_def:
    f = args.output_graphdef_file 
    tf.train.write_graph(sess.graph.as_graph_def(), output_fld, f, as_text=True)
    print('saved the graph definition in ascii format at: ', str(Path(output_fld) / f))


# convert variables to constants and save

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
if args.quantize:
    from tensorflow.tools.graph_transforms import TransformGraph
    transforms = ["quantize_weights", "quantize_nodes"]
    transformed_graph_def = TransformGraph(sess.graph.as_graph_def(), [], pred_node_names, transforms)
    constant_graph = graph_util.convert_variables_to_constants(sess, transformed_graph_def, pred_node_names)
else:
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)    
graph_io.write_graph(constant_graph, output_fld, args.output_model_file, as_text=False)
print('saved the freezed graph (ready for inference) at: ', str(Path(output_fld) / args.output_model_file))

