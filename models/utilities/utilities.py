#!/usr/bin/env python

"""utilities: The place where all the accessories functions are dumped.

Functions:
    huber_loss: Return L1 error if absolute error is less than clip_value, else
                return L2 error.
    clipped_error: Call huber_loss with default clip_value to 1.0.
"""

from argparse import ArgumentParser
import numpy as np
from keras import backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import *
import tensorflow as tf

__author__ = "Victor Neves"
__license__ = "MIT"
__maintainer__ = "Victor Neves"
__email__ = "victorneves478@gmail.com"

def huber_loss(y_true, y_pred, clip_value):
	# Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
	# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
	# for details.
	assert clip_value > 0.

	x = y_true - y_pred
	if np.isinf(clip_value):
		# Spacial case for infinity since Tensorflow does have problems
		# if we compare `K.abs(x) < np.inf`.
		return .5 * K.square(x)

	condition = K.abs(x) < clip_value
	squared_loss = .5 * K.square(x)
	linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
	if K.backend() == 'tensorflow':
		if hasattr(tf, 'select'):
			return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
		else:
			return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
	elif K.backend() == 'theano':
		from theano import tensor as T
		return T.switch(condition, squared_loss, linear_loss)
	else:
		raise RuntimeError('Unknown backend "{}".'.format(K.backend()))

def clipped_error(y_true, y_pred):
	return K.mean(huber_loss(y_true, y_pred, clip_value = 1.), axis = -1)

class HandleArguments:
        """Handle arguments provided in the command line when executing the model.

        Attributes:
            args: arguments parsed in the command line.
            status_load: a flag for usage of --load argument.
            status_visual: a flag for usage of --visual argument.
        """
        def __init__(self):
            self.parser = ArgumentParser() # Receive arguments
            self.parser.add_argument("-l", "--load", help = "load a previously trained model. the argument is the filename", required = False, default = "")
            self.parser.add_argument("-v", "--visual", help = "define board size", required = False, action = 'store_true')
            self.parser.add_argument("-ls", "--local_state", help = "define board size", required = False, action = 'store_true')
            self.parser.add_argument("-g", "--board_size", help = "define board size", required = False, default = 10, type = int)
            self.parser.add_argument("-nf", "--nb_frames", help = "define board size", required = False, default = 4, type = int)
            self.parser.add_argument("-na", "--nb_actions", help = "define board size", required = False, default = 5, type = int)

            self.args = self.parser.parse_args()
            self.status_load = False
            self.status_visual = False
            self.local_state = False

            if self.args.load:
                script_dir = path.dirname(__file__) # Absolute dir the script is in
                abs_file_path = path.join(script_dir, self.args.load)
                model = load_model(abs_file_path)

                self.status_load = True

            if self.args.visual:
                self.status_visual = True

            if self.args.local_state:
                self.local_state = True

def CNN1(optimizer, loss, stack, input_size, output_size):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation = 'relu', input_shape = (stack,
                                                                     input_size,
                                                                     input_size)))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(output_size))
    model.compile(optimizer = optimizer, loss = loss)

    return model

def CNN2(optimizer, loss, stack, input_size, output_size, min_neurons = 16,
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

def CNN3(optimizer, loss, stack, input_size, output_size, drop_prob = 0.1):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation = 'relu',input_shape = (stack,
                                                                    input_size,
                                                                    input_size)))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(drop_prob))
    model.add(Dense(output_size))
    model.compile(optimizer = optimizer, loss = loss)

    return model

def CNN_DUELING(optimizer, loss, stack, input_size, output_size):
    inputs = Input(shape = (stack, input_size, input_size))
    net = Conv2D(16, (3, 3), activation = 'relu')(inputs)
    net = Conv2D(32, (3, 3), activation = 'relu')(net)
    net = Flatten()(net)
    advt = Dense(256, activation = 'relu')(net)
    advt = Dense(output_size)(advt)
    value = Dense(256, activation = 'relu')(net)
    value = Dense(1)(value)
    # now to combine the two streams
    advt = Lambda(lambda advt: advt - tf.reduce_mean(advt, axis = -1,
                                                     keep_dims = True))(advt)
    value = Lambda(lambda value: tf.tile(value, [1, output_size]))(value)
    final = Add()([value, advt])
    model = Model(inputs = inputs, outputs = final)
    model.compile(optimizer = optimizer, loss = loss)

    return model
