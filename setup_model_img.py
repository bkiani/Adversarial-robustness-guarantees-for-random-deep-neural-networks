## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras.initializers import VarianceScaling
import keras

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def my_sparse_categorical_crossentropy(y_true, y_pred):
    return keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

class img_data_generator:
    def __init__(self, num_labels = 2, image_size = 10, bounds = [0.,1.], num_channels = 1):
        self.num_labels = num_labels
        self.image_size = image_size
        self.bounds = bounds
        self.num_channels = num_channels

    def generate_images(self, num_images):
        return np.random.uniform(   low = self.bounds[0], high = self.bounds[1],
                                    size = (num_images, self.image_size, self.image_size, self.num_channels))

    def generate_labels(self, num_images):
        int_labels = self.get_int_labels(num_images)
        one_hot_targets = np.eye(self.num_labels)[int_labels]
        return one_hot_targets

    def get_int_labels(self, num_images):
        return np.random.randint(self.num_labels, size = num_images)

    def generate_dataset(self, num_images):
        return self.generate_images(num_images), self.get_int_labels(num_images)



class FF_simple_2_layer_img_model:
    def __init__(self, session=None, use_logits=True,
        num_labels = 2, image_size = 10):
        self.num_channels = 1
        self.image_size = image_size
        self.num_labels = num_labels

        model = Sequential()
        model.add(Flatten())
        model.add(Dense(image_size))
        model.add(Activation('relu'))
        model.add(Dense(image_size))
        model.add(Activation('relu'))
        model.add(Dense(num_labels))
        # output log probability, used for black-box attack
        if not use_logits:
            model.add(Activation('softmax'))
        if restore:
            model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)

class FF_img_model:
    def __init__(self, restore = False, use_logits=True,
                        num_labels = 2, image_size = 10,
                        hidden_shape = [100,100],
                        A_type = 'relu',
                        init_type = 'VarianceScaling',
                        b_init_type = 'zeros',
                        include_bias = False,
                        var_scale_val = 2.0,
                        num_channels = 1):
        self.num_channels = 1
        self.image_size = image_size
        self.num_labels = num_labels

        model = Sequential()
        model.add(Flatten())

        net_shape = [image_size*image_size*num_channels]+hidden_shape

        if init_type == 'VarianceScaling':
            init_type = VarianceScaling(var_scale_val,
                mode = 'fan_in', distribution = 'normal')

        for layer_i in range(len(net_shape)-1):
            model.add(Dense(net_shape[layer_i+1],
                input_shape = (net_shape[layer_i],),
                activation = A_type,
                use_bias = include_bias,
                kernel_initializer=init_type,
                bias_initializer=b_init_type))

        model.add(Dense(num_labels,
            use_bias = include_bias,
            bias_initializer=b_init_type,
            kernel_initializer=init_type))

        # output log probability, used for black-box attack
        if not use_logits:
            model.add(Activation('softmax'))


        self.model = model

    def predict(self, data):
        return self.model.predict(data)

    def train(self,  X_train, y_train,
                     X_val = None, y_val = None,
                     n_epochs = 25):
        self.model.compile(optimizer=keras.optimizers.Adam(lr = 0.0001),  # Optimizer
              # Loss function to minimize
              loss=keras.losses.sparse_categorical_crossentropy,
              # List of metrics to monitor
              metrics=[keras.metrics.sparse_categorical_accuracy])

        if X_val is None:
            X_val = X_train
            y_val = y_train

        self.model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=n_epochs,
                    # We pass someaaa validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(X_val, y_val),
                    verbose = 1)



class VGG_img_model:
    def __init__(self, restore = None, use_logits=True,
                        num_labels = 2, image_size = 64,
                        init_type = 'VarianceScaling',
                        var_scale_val = 2.0):
        self.num_channels = 1
        self.image_size = image_size
        self.num_labels = num_labels

        if init_type == 'VarianceScaling':
            init_type = VarianceScaling(var_scale_val,
                mode = 'fan_in', distribution = 'normal')
        elif init_type == 'glorot_uniform':
            pass


        model = Sequential([
            Conv2D(64, (3, 3), input_shape=(image_size,image_size,3), padding='same', activation='relu', kernel_initializer=init_type),
            Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=init_type),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=init_type),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=init_type,),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=init_type,),
            Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=init_type),
            Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=init_type),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=init_type),
            Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=init_type),
            Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=init_type),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=init_type),
            Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=init_type),
            Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=init_type),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        ])

        model.add(Flatten())
        # model.add(Dense(4096, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(4096, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(num_labels))

        if not use_logits:
            model.add(Activation('softmax'))
        if restore:
            model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model(data)

class simple_conv_model:
    def __init__(self, restore = None, use_logits=True,
                        num_labels = 2, image_size = 64,
                        init_type = 'VarianceScaling',
                        var_scale_val = 0.02):
        self.image_size = image_size
        self.num_labels = num_labels

        if init_type == 'VarianceScaling':
            init_type = VarianceScaling(var_scale_val,
                mode = 'fan_in', distribution = 'normal')
        elif init_type == 'glorot_uniform':
            pass


        model = Sequential([
            Conv2D(128, (3, 3), input_shape=(image_size,image_size,3), padding='same', activation='relu', kernel_initializer=init_type),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=init_type),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=init_type),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=init_type,),
        ])

        model.add(Flatten())
        # model.add(Dense(1024, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(4096, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(num_labels))

        if not use_logits:
            model.add(Activation('softmax'))
        # if restore is not None:
        #     model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model.predict(data)

    def train(self,  X_train, y_train,
                     X_val = None, y_val = None,
                     n_epochs = 25):
        self.model.compile(optimizer=keras.optimizers.Adam(lr = 0.0001),  # Optimizer
              # Loss function to minimize
              loss=keras.losses.sparse_categorical_crossentropy,
              # List of metrics to monitor
              metrics=[keras.metrics.sparse_categorical_accuracy])

        if X_val is None:
            X_val = X_train
            y_val = y_train

        self.model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=n_epochs,
                    # We pass someaaa validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(X_val, y_val))


class le_net_model:
    def __init__(self,  image_size = 64,
                        use_logits=True,
                        num_labels = 2,
                        init_type = 'VarianceScaling',
                        var_scale_val = 2.0, num_channels = 3):
        self.num_channels = 1
        self.image_size = image_size
        self.num_labels = num_labels

        if init_type == 'VarianceScaling':
            init_type = VarianceScaling(var_scale_val,
                mode = 'fan_in', distribution = 'normal')
        elif init_type == 'glorot_uniform':
            pass


        model = Sequential([
            Conv2D(128, (3, 3), input_shape=(image_size,image_size,num_channels), padding='same', activation='relu', kernel_initializer=init_type, use_bias = False),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=init_type, use_bias = False),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=init_type, use_bias = False),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=init_type, use_bias = False),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        ])

        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer=init_type, use_bias = False))
        model.add(Dense(512, activation='relu', kernel_initializer=init_type, use_bias = False))
        model.add(Dense(num_labels))

        if not use_logits:
            model.add(Activation('softmax'))

        self.model = model

    def predict(self, data):
        return self.model.predict(data)

    def train(self,  X_train, y_train,
                     X_val = None, y_val = None,
                     n_epochs = 25,
                     use_categorical = False):
        if use_categorical:
            self.model.compile(optimizer=keras.optimizers.Adam(lr = 0.0001),  # Optimizer
                  # Loss function to minimize
                  loss=keras.losses.categorical_crossentropy,
                  # List of metrics to monitor
                  metrics=[keras.metrics.categorical_accuracy])
        else: 
            self.model.compile(optimizer=keras.optimizers.Adam(lr = 0.0001),  # Optimizer
                  # Loss function to minimize
                  loss=keras.losses.sparse_categorical_crossentropy,
                  # List of metrics to monitor
                  metrics=[keras.metrics.sparse_categorical_accuracy])

        if X_val is None:
            X_val = X_train
            y_val = y_train

        self.model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=n_epochs,
                    # We pass someaaa validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(X_val, y_val))

class le_net_cut_model:
    def __init__(self,  image_size = 64,
                        use_logits=True,
                        num_labels = 2,
                        init_type = 'VarianceScaling',
                        var_scale_val = 2.0, num_channels = 3):
        self.num_channels = 1
        self.image_size = image_size
        self.num_labels = num_labels

        if init_type == 'VarianceScaling':
            init_type = VarianceScaling(var_scale_val,
                mode = 'fan_in', distribution = 'normal')
        elif init_type == 'glorot_uniform':
            pass


        model = Sequential([
            Conv2D(128, (3, 3), input_shape=(image_size,image_size,num_channels), padding='same', activation='relu', kernel_initializer=init_type, use_bias = False),
            Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=init_type, use_bias = False),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
            # Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=init_type, use_bias = False),
            # Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=init_type, use_bias = False),
            AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        ])

        model.add(Flatten())
        # model.add(Dense(512, activation='relu', kernel_initializer=init_type, use_bias = False))
        model.add(Dense(512, activation='relu', kernel_initializer=init_type, use_bias = False))
        model.add(Dense(num_labels))

        if not use_logits:
            model.add(Activation('softmax'))

        self.model = model

    def predict(self, data):
        return self.model.predict(data)

    def train(self,  X_train, y_train,
                     X_val = None, y_val = None,
                     n_epochs = 25):
        self.model.compile(optimizer=keras.optimizers.Adam(lr = 0.0001),  # Optimizer
              # Loss function to minimize
              loss=keras.losses.sparse_categorical_crossentropy,
              # List of metrics to monitor
              metrics=[keras.metrics.sparse_categorical_accuracy])

        if X_val is None:
            X_val = X_train
            y_val = y_train

        self.model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=n_epochs,
                    # We pass someaaa validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(X_val, y_val))