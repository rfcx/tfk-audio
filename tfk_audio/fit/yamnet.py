# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Core model definition of YAMNet."""

import csv

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

def _batch_norm(name, params):
    def _bn_layer(layer_input):
        return layers.BatchNormalization(
            name=name,
            center=params.batchnorm_center,
            scale=params.batchnorm_scale,
            epsilon=params.batchnorm_epsilon)(layer_input)
    return _bn_layer


def _conv(name, kernel, stride, filters, params):
    def _conv_layer(layer_input):
        output = layers.Conv2D(name='{}/conv'.format(name),
                                                     filters=filters,
                                                     kernel_size=kernel,
                                                     strides=stride,
                                                     padding=params.conv_padding,
                                                     use_bias=False,
                                                     activation=None)(layer_input)
        output = _batch_norm('{}/conv/bn'.format(name), params)(output)
        output = layers.ReLU(name='{}/relu'.format(name))(output)
        return output
    return _conv_layer


def _separable_conv(name, kernel, stride, filters, params):
    def _separable_conv_layer(layer_input):
        output = layers.DepthwiseConv2D(name='{}/depthwise_conv'.format(name),
                                                                        kernel_size=kernel,
                                                                        strides=stride,
                                                                        depth_multiplier=1,
                                                                        padding=params.conv_padding,
                                                                        use_bias=False,
                                                                        activation=None)(layer_input)
        output = _batch_norm('{}/depthwise_conv/bn'.format(name), params)(output)
        output = layers.ReLU(name='{}/depthwise_conv/relu'.format(name))(output)
        output = layers.Conv2D(name='{}/pointwise_conv'.format(name),
                                                     filters=filters,
                                                     kernel_size=(1, 1),
                                                     strides=1,
                                                     padding=params.conv_padding,
                                                     use_bias=False,
                                                     activation=None)(output)
        output = _batch_norm('{}/pointwise_conv/bn'.format(name), params)(output)
        output = layers.ReLU(name='{}/pointwise_conv/relu'.format(name))(output)
        return output
    return _separable_conv_layer


_YAMNET_LAYER_DEFS = [
        # (layer_function, kernel, stride, num_filters)
        (_conv,                    [3, 3], 2,     32),
        (_separable_conv, [3, 3], 1,     64),
        (_separable_conv, [3, 3], 2,    128),
        (_separable_conv, [3, 3], 1,    128),
        (_separable_conv, [3, 3], 2,    256),
        (_separable_conv, [3, 3], 1,    256),
        (_separable_conv, [3, 3], 2,    512),
        (_separable_conv, [3, 3], 1,    512),
        (_separable_conv, [3, 3], 1,    512),
        (_separable_conv, [3, 3], 1,    512),
        (_separable_conv, [3, 3], 1,    512),
        (_separable_conv, [3, 3], 1,    512),
        (_separable_conv, [3, 3], 2, 1024),
        (_separable_conv, [3, 3], 1, 1024)
]


def yamnet(features, params):
    """Define the core YAMNet mode in Keras."""
    net = layers.Reshape(
            (params.patch_frames, params.patch_bands, 1),
            input_shape=(params.patch_frames, params.patch_bands))(features)
    for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
        net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters, params)(net)
    embeddings = layers.GlobalAveragePooling2D()(net)
    logits = layers.Dense(units=params.num_classes, use_bias=True)(embeddings)
    predictions = layers.Activation(activation=params.classifier_activation)(logits)
    return predictions, embeddings



# class YAMNet(tf.keras.Model):
#     def __init__(self, config):    
#         super().__init__()
        
#         self._config = config
        
#         class params:
#             patch_bands = config['spec']['mel_bands']
#             patch_frames = int(config['audio']['sample_seconds']/config['spec']['stft_hop_seconds']-(config['spec']['stft_window_seconds']/config['spec']['stft_hop_seconds'])+1)
#             num_classes = config['fit']['output_nodes']
#             classifier_activation = config['fit']['output_activation']
#             # rest of parameters left to YAMNet defaults
#             conv_padding = 'same'
#             batchnorm_center = True
#             batchnorm_scale = False
#             batchnorm_epsilon = 1e-4
            
#         self.conv1 = yamnet_conv_model(params)
#         self.dense1 = layers.Dense(64, activation='relu')
#         self.dropout1 = layers.Dropout(config['fit']['dropout'])
#         self.dense2 = layers.Dense(config['fit']['output_nodes'], activation=config['fit']['output_activation'])
#         self.build(input_shape=self.conv1.input_shape)
        
#     def call(self, inputs, training=False):
#         x = self.conv1(inputs)
#         x = self.dense1(x)
#         x = self.dropout1(x)
#         return self.dense2(x)
    
#     def summary(self):
#         x = layers.Input(self.conv1.input_shape[1:])
#         model = tf.keras.Model(inputs=x, outputs=self.call(x))
#         return model.summary()
    

# def yamnet_conv_model(params):
#     specs = layers.Input(batch_shape=(None, params.patch_frames, params.patch_bands), dtype=tf.float32)
#     _, embeddings = yamnet.yamnet(specs, params)
#     model = tf.keras.Model(inputs=specs, outputs=embeddings)
#     return model
