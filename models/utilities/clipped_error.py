#!/usr/bin/env python

"""clipped_error: L1 for errors < clip_value else L2 error.

Functions:
    huber_loss: Return L1 error if absolute error is less than clip_value, else
                return L2 error.
    clipped_error: Call huber_loss with default clip_value to 1.0.
"""

import numpy as np
from keras import backend as K
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
