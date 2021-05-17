from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.initializers import VarianceScaling
import numpy as np
import os

# taken from https://keras.io/examples/cifar10_resnet/


def resnet_layer(inputs,
                 num_filters=32,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=False,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


class resnet_v1:
    def __init__(  self,input_shape, depth, num_classes=2,
                    init_type = 'VarianceScaling', var_scale_val = 2.0,
                    restore = None, use_logits=True, global_avg_pool = False,
                    include_conv_at_start = False):
        """ResNet Version 1 Model builder [a]

        Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
        Last ReLU is after the shortcut connection.
        At the beginning of each stage, the feature map size is halved (downsampled)
        by a convolutional layer with strides=2, while the number of filters is
        doubled. Within each stage, the layers have the same number filters and the
        same number of filters.
        Features maps sizes:
        stage 0: 32x32, 16
        stage 1: 16x16, 32
        stage 2:  8x8,  64
        The Number of parameters is approx the same as Table 6 of [a]:
        ResNet20 0.27M
        ResNet32 0.46M
        ResNet44 0.66M
        ResNet56 0.85M
        ResNet110 1.7M

        # Arguments
            input_shape (tensor): shape of input image tensor
            depth (int): number of core convolutional layers
            num_classes (int): number of classes (CIFAR10 has 10)

        # Returns
            model (Model): Keras model instance
        """
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        # Start model definition.

        if init_type == 'VarianceScaling':
            init_type = VarianceScaling(var_scale_val,
                mode = 'fan_in', distribution = 'normal')
        elif init_type == 'glorot_uniform':
            pass

        num_filters = 32    
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=input_shape)
        if include_conv_at_start:
            conv_start = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal')
            x = conv_start(inputs)
            x = Activation('relu')(x)
            x = resnet_layer(inputs=x)
        else:
          x = resnet_layer(inputs=inputs)


        
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        if global_avg_pool:
            x = AveragePooling2D(pool_size=int(input_shape[0]/4))(x)
        y = Flatten()(x)
        if use_logits:
            outputs = Dense(num_classes,
                            activation='linear',
                            kernel_initializer=init_type)(y)
        else:
            outputs = Dense(num_classes,
                            activation='softmax',
                            kernel_initializer=init_type)(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)

        
        if restore:
            model.load_weights(restore)

        self.model = model

    def predict(self, data):
        return self.model.predict(data)

    def train(self,  X_train, y_train,
                     X_val = None, y_val = None,
                     n_epochs = 10,
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
                    validation_data=(X_val, y_val))