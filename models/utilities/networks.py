"""THIS"""
#!/usr/bin/env python

""" Needs update!
"""

import numpy as np
import tensorflow as tf
try:
    from keras.optimizers import RMSprop, Nadam
    from keras.models import Sequential, load_model, Model
    from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Flatten,\
                             Input, Lambda, Add
    from keras import backend as K

    K.set_image_dim_ordering('th')
except ImportError:
    from tensorflow.keras.optimizers import RMSprop, Nadam
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D,\
                                        Flatten, Input, Lambda, Add

from .noisy_dense import NoisyDenseFG, NoisyDenseIG

__author__ = "Victor Neves"
__license__ = "MIT"
__maintainer__ = "Victor Neves"
__email__ = "victorneves478@gmail.com"

DENSES = {'dense': Dense,
          'noisy_dense_fg': NoisyDenseFG,
          'noisy_dense_ig': NoisyDenseIG}

def select_optimizer(optimizer):
    assert optimizer in {'Nadam', 'RMSprop'}, "Optimizer should be RMSprop or Nadam."

    if optimizer == 'Nadam':
        optimizer = Nadam()
    else:
        optimizer = RMSprop()

    return optimizer

def select_error(error):
    assert type(error) is str, "Should use string to select error."

    if error == 'huber_loss':
        error = tf.losses.huber_loss

    return error

def CNN1(inputs):
    net = Conv2D(16, (3, 3), activation = 'relu')(inputs)
    net = Conv2D(32, (3, 3), activation = 'relu')(net)

    return model

def CNN2(inputs):
    net = Conv2D(16, (3, 3), activation = 'relu')(inputs)
    net = Conv2D(32, (3, 3), activation = 'relu')(net)
    net = Conv2D(32, (3, 3), activation = 'relu')(net)
    net = Flatten()(net)

    return model

def CNN3(inputs):
    """From @Kaixhin implementation's of the Rainbow paper."""
    net = Conv2D(32, (3, 3), activation = 'relu')(inputs)
    net = Conv2D(64, (2, 2), activation = 'relu')(net)
    net = Conv2D(64, (1, 1), activation = 'relu')(net)
    net = Flatten()(net)

    return net

def create_cnn(cnn, inputs):
    if cnn == "CNN1":
        net = CNN1(inputs)
    elif cnn == "CNN2":
        net = CNN2(inputs)
    else:
        net = CNN3(inputs)

    return net

def create_model(optimizer, loss, stack, input_size, output_size,
                 dueling = False, cnn = "CNN3", dense_type = "dense"):
    optimizer = select_optimizer(optimizer)
    loss = select_error(loss)
    inputs = Input(shape = (input_size, input_size, stack))
    net = create_cnn(cnn, inputs)

    if dueling:
        advt = DENSES[dense_type](3136, activation = 'relu')(net)
        advt = DENSES[dense_type](output_size)(advt)
        value = DENSES[dense_type](3136, activation = 'relu')(net)
        value = DENSES[dense_type](1)(value)

        # now to combine the two streams
        advt = Lambda(lambda advt: advt - tf.reduce_mean(advt, axis = -1,
                                                         keepdims = True))(advt)
        value = Lambda(lambda value: tf.tile(value, [1, output_size]))(value)
        final = Add()([value, advt])
    else:
        final = DENSES[dense_type](3136, activation = 'relu')(net)
        final = DENSES[dense_type](output_size)(final)

    model = Model(inputs = inputs, outputs = final)
    model.compile(optimizer = optimizer, loss = loss)

    return model
