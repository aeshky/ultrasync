"""
Date: Oct 2018
Author: Aciel Eshky

Adapted from: https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py

"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, BatchNormalization, Dropout, Conv2D, Activation, MaxPooling2D, Softmax
from keras import regularizers
from keras import backend as K
from keras.layers import Lambda


class TwoStreamNetwork:

    def __init__(self, config):

        self.config = config
        self.output_size = int(self.config["output_size"])
        self.L2_reg = self.config["L2_reg"]
        self.model_type = self.config["model_type"]
        self.input_shape_visual = self.config['dim_visual']
        self.input_shape_audio = self.config['dim_audio']

        self.kernel_size_visual = self.config["kernel_size_visual"]
        self.conv_strides_visual = self.config["conv_strides_visual"]
        self.pool_size_visual = self.config["pool_size_visual"]
        self.pool_strides_visual = self.config["pool_strides_visual"]
        self.shrinking_size_visual = self.config["shrinking_size_visual"]
        self.shrinking_strides_visual = self.config["shrinking_strides_visual"]

        self.kernel_size_audio = self.config["kernel_size_audio"]
        self.conv_strides_audio = self.config["conv_strides_audio"]
        self.pool_size_audio = self.config["pool_size_audio"]
        self.pool_strides_audio = self.config["pool_strides_audio"]
        self.shrinking_size_audio = self.config["shrinking_size_audio"]
        self.shrinking_strides_audio = self.config["shrinking_strides_audio"]
        self.five_layers = self.config["five_layers"]

        self.dropout = self.config["dropout"]

        assert self.model_type in ["SyncNet"]

        base_network_visual = self.__create_base_network_visual(self.input_shape_visual)
        base_network_audio = self.__create_base_network_audio(self.input_shape_audio)

        input_visual = Input(shape=self.input_shape_visual)
        input_audio = Input(shape=self.input_shape_audio)

        processed_a = base_network_visual(input_visual)
        processed_b = base_network_audio(input_audio)

        self.distance = Lambda(euclidean_distance, output_shape=get_output_shape)([processed_a, processed_b])

        self.model = Model([input_visual, input_audio], self.distance)

        if config["loss_function"] == "binary_crossentropy":
            self.combine_layers([input_visual, input_audio], self.distance)

        print(self.model.summary())

    def combine_layers(self, combined_input, distance):
        print(distance)
        x = Dense(1)(distance)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(combined_input, x)

    def __create_base_network_visual(self, input_shape):
        """ Base network for visual input """
        visual_input = Input(shape=input_shape)
        x = visual_input

        if self.model_type == "SyncNet":

            # input shape (5, 63, 138)
            # input_shape = (input_shape[0], input_shape[2], input_shape[1])
            # (5, 138, 63) (channels, height, width)

            # conv1_visual

            x = Conv2D(self.output_size // 2, kernel_size=self.kernel_size_visual[0], strides=self.conv_strides_visual,
                       padding='valid', name='conv1_visual', data_format="channels_first")(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # bn1_visual
            x = BatchNormalization(name='bn1_visual')(x)

            # relu1_visual
            x = Activation('relu', name='relu1_visual')(x)

            # pool1_visual
            x = MaxPooling2D(pool_size=self.shrinking_size_visual, strides=self.shrinking_strides_visual,
                             padding='valid', name='pool1_shriking_visual', data_format="channels_first")(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # conv2_visual
            x = Conv2D(self.output_size, kernel_size=self.kernel_size_visual[1], strides=self.conv_strides_visual,
                       padding='valid', name='conv2_visual', data_format="channels_first")(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # bn2_visual
            x = BatchNormalization(name='bn2_visual')(x)

            # relu2_visual
            x = Activation('relu', name='relu2_visual')(x)

            # pool2_visual
            x = MaxPooling2D(pool_size=self.pool_size_visual, strides=self.pool_strides_visual,
                             padding='valid', name='pool2_visual', data_format="channels_first")(x)
            print("shape of", x.name, "output:", x._keras_shape)

            if self.five_layers:
                # conv3_visual
                x = Conv2D(self.output_size * 2, kernel_size=self.kernel_size_visual[2],
                           strides=self.conv_strides_visual,
                           padding='valid', name='conv3_visual')(x)

                # bn3_visual
                x = BatchNormalization(name='bn3_visual')(x)

                # relu3_visual
                x = Activation('relu', name='relu3_visual')(x)

                # conv4_visual
                x = Conv2D(self.output_size * 2, kernel_size=self.kernel_size_visual[2],
                           strides=self.conv_strides_visual,
                           padding='valid', name='conv4_visual')(x)

                # bn4_visual
                x = BatchNormalization(name='bn4_visual')(x)

                # relu4_visual
                x = Activation('relu', name='relu4_visual')(x)

            # conv5_visual
            x = Conv2D(self.output_size * 2, kernel_size=self.kernel_size_visual[2], strides=self.conv_strides_visual,
                       padding='valid', name='conv5_visual', data_format="channels_first")(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # bn5_visual
            x = BatchNormalization(name='bn5_visual')(x)

            # relu5_visual
            x = Activation('relu', name='relu5_visual')(x)

            # pool5_visual
            x = MaxPooling2D(pool_size=self.pool_size_visual, strides=self.pool_strides_visual,
                             padding='valid', name='pool5_visual', data_format="channels_first")(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # fc6_visual
            x = Flatten(name='flatten_visual')(x)
            x = Dense(self.output_size, name='fc6_visual')(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # bn6_visual
            x = BatchNormalization(name='bn6_visual')(x)

            # relu6_visual
            x = Activation('relu', name='relu6_visual')(x)

            # fc7_visual
            x = Dense(self.output_size, name='fc7_visual')(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # bn7_visual
            x = BatchNormalization(name='bn7_visual')(x)

            # relu7_visual
            x = Activation('relu', name='relu7_visual')(x)

        return Model(visual_input, x)

    def __create_base_network_audio(self, input_shape):
        """ Base network for audio input """
        audio_input = Input(shape=input_shape)
        x = audio_input

        if self.model_type == "SyncNet":

            # input = (1, 20, 12)
            # input_shape = (input_shape[0], input_shape[2], input_shape[1])
            #  (1, 12, 20) (channels, height, width)

            # conv1_audio
            x = Conv2D(self.output_size // 2, kernel_size=self.kernel_size_audio[0], strides=self.conv_strides_audio,
                       padding='same', name='conv1_audio', data_format="channels_first")(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # bn1_audio
            x = BatchNormalization(name='bn1_audio')(x)

            # relu1_audio
            x = Activation('relu', name='relu1_audio')(x)

            # conv2_audio
            x = Conv2D(self.output_size, kernel_size=self.kernel_size_audio[1], strides=self.conv_strides_audio,
                       padding='same', name='conv2_audio', data_format="channels_first")(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # bn2_audio
            x = BatchNormalization(name='bn2_audio')(x)

            # relu2_audio
            x = Activation('relu', name='relu2_audio')(x)

            # pool2_audio
            x = MaxPooling2D(pool_size=self.shrinking_size_audio, strides=self.shrinking_strides_audio,
                             padding='valid', name='pool2_shriking_audio', data_format="channels_first")(x)
            print("shape of", x.name, "output:", x._keras_shape)

            if self.five_layers:
                # conv3_audio
                x = Conv2D(self.output_size * 2, kernel_size=self.kernel_size_audio[2], strides=self.conv_strides_audio,
                           padding='same', name='conv3_audio')(x)

                # bn3_audio
                x = BatchNormalization(name='bn3_audio')(x)

                # relu3_audio
                x = Activation('relu', name='relu3_audio')(x)

                # conv4_audio
                x = Conv2D(self.output_size * 2, kernel_size=self.kernel_size_audio[2], strides=self.conv_strides_audio,
                           padding='same', name='conv4_audio')(x)

                # bn4_audio
                x = BatchNormalization(name='bn4_audio')(x)

                # relu4_audio
                x = Activation('relu', name='relu4_audio')(x)

            # conv5_audio
            x = Conv2D(self.output_size * 2, kernel_size=self.kernel_size_audio[2], strides=self.conv_strides_audio,
                       padding='same', name='conv5_audio', data_format="channels_first")(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # bn5_audio
            x = BatchNormalization(name='bn5_audio')(x)

            # relu5_audio
            x = Activation('relu', name='relu5_audio')(x)

            # pool5_audio
            x = MaxPooling2D(pool_size=self.pool_size_audio, strides=self.pool_strides_audio,
                             padding='valid', name='pool5_audio', data_format="channels_first")(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # fc6_audio
            x = Flatten(name='flatten_audio')(x)
            x = Dense(self.output_size, name='fc6_audio')(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # bn6_audio
            x = BatchNormalization(name='bn6_audio')(x)

            # relu6_audio
            x = Activation('relu', name='relu6_audio')(x)

            # fc7_audio
            x = Dense(self.output_size, name='fc7_audio')(x)
            print("shape of", x.name, "output:", x._keras_shape)

            # bn7_audio
            x = BatchNormalization(name='bn7_audio')(x)

            # relu7_audio
            x = Activation('relu', name='relu7_audio')(x)

        return Model(audio_input, x)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def get_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    """ Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf """
    margin = 1
    return K.mean(y_true * K.square(y_pred) +  # postive condition
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))  # negative condition


def compute_accuracy(y_true, y_pred):
    """ Compute classification accuracy with a fixed threshold on distances. """
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    """ Compute classification accuracy with a fixed threshold on distances. """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
