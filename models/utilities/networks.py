#!/usr/bin/env python

""" Needs update!
"""

import numpy as np
from keras.models import Sequential, load_model, Model
from keras.layers import *
import tensorflow as tf

__author__ = "Victor Neves"
__license__ = "MIT"
__maintainer__ = "Victor Neves"
__email__ = "victorneves478@gmail.com"

def weird_CNN(optimizer, loss, stack, input_size, output_size, min_neurons = 16,
         max_neurons = 128, kernel_size = (3,3), layers = 4):
    # INPUTS
    # size     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN

    # Define hyperparamters
    MIN_NEURONS = min_neurons
    MAX_NEURONS = max_neurons
    KERNEL = kernel_size
    n_layers = layers

    # Determine the # of neurons in each convolutional layer
    steps = np.floor(MAX_NEURONS / (n_layers + 1))
    neurons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
    neurons = neurons.astype(np.int32)

    # Define a model
    model = Sequential()

    # Add convolutional layers
    for i in range(0, n_layers):
        if i == 0:
            model.add(Conv2D(neurons[i], KERNEL, input_shape = (stack,
                                                                input_size,
                                                                input_size)))
        else:
            model.add(Conv2D(neurons[i], KERNEL))

        model.add(Activation('relu'))

    # Add max pooling layer
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(MAX_NEURONS * 4))
    model.add(Activation('relu'))

    # Add output layer
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))

    # Compile the model
    model.compile(loss = loss, optimizer = optimizer)

    return model

def CNN1(inputs, stack, input_size):
    net = Conv2D(16, (3, 3), activation = 'relu')(inputs)
    net = Conv2D(32, (3, 3), activation = 'relu')(net)

    return model

def CNN2(inputs, stack, input_size):
    net = Conv2D(16, (3, 3), activation = 'relu')(inputs)
    net = Conv2D(32, (3, 3), activation = 'relu')(net)
    net = Conv2D(32, (3, 3), activation = 'relu')(net)
    net = MaxPooling2D(pool_size = (2, 2))(net)
    net = Flatten()(net)

    return model

def CNN3(inputs, stack, input_size):
    """From @Kaixhin implementation's of the Rainbow paper."""
    net = Conv2D(32, (4, 4), activation = 'relu')(inputs)
    net = Conv2D(64, (2, 2), activation = 'relu')(net)
    net = Conv2D(64, (2, 2), activation = 'relu')(net)
    net = Flatten()(net)

    return net

def create_cnn(cnn, inputs, stack, input_size):
    if cnn == "CNN1":
        net = CNN1(inputs, stack, input_size)
    elif cnn == "CNN2":
        net = CNN2(inputs, stack, input_size)
    else:
        net = CNN3(inputs, stack, input_size)

    return net

def create_model(optimizer, loss, stack, input_size, output_size,
                  dueling = False, cnn = "CNN3"):
    inputs = Input(shape = (stack, input_size, input_size))
    net = create_cnn(cnn, inputs, stack, input_size)

    if dueling:
        advt = Dense(3136, activation = 'relu')(net)
        advt = Dense(output_size)(advt)
        value = Dense(3136, activation = 'relu')(net)
        value = Dense(1)(value)

        # now to combine the two streams
        advt = Lambda(lambda advt: advt - tf.reduce_mean(advt, axis = -1,
                                                         keepdims = True))(advt)
        value = Lambda(lambda value: tf.tile(value, [1, output_size]))(value)
        final = Add()([value, advt])
    else:
        final = Dense(3136, activation = 'relu')(net)
        final = Dense(output_size)(final)

    model = Model(inputs = inputs, outputs = final)
    model.compile(optimizer = optimizer, loss = loss)

    return model
