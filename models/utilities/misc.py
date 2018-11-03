#!/usr/bin/env python
import numpy as np
from argparse import ArgumentParser

from keras import backend as K
import keras.optimizers as optimizers
import tensorflow as tf

__author__ = "Victor Neves"
__license__ = "MIT"
__maintainer__ = "Victor Neves"
__email__ = "victorneves478@gmail.com"

class HandleArguments:
        """Handle arguments provided in the command line when executing the model.

        Attributes:
            args: arguments parsed in the command line.
            status_load: a flag for usage of --load argument.
            status_visual: a flag for usage of --visual argument.

            NEED UPDATE!
        """
        def __init__(self):
            self.parser = ArgumentParser() # Receive arguments
            self.parser.add_argument("-l", "--load", help = "load a previously trained model. the argument is the filename", required = False, default = "")
            self.parser.add_argument("-v", "--visual", help = "define board size", required = False, action = 'store_true')
            self.parser.add_argument("-du", "--dueling", help = "use dueling DQN", required = False, action = 'store_true')
            self.parser.add_argument("-do", "--double", help = "use double DQN", required = False, action = 'store_true')
            self.parser.add_argument("-p", "--per", help = "use Prioritized Experience Replay", required = False, action = 'store_true')
            self.parser.add_argument("-ls", "--local_state", help = "define board size", required = False, action = 'store_true')
            self.parser.add_argument("-g", "--board_size", help = "define board size", required = False, default = 10, type = int)
            self.parser.add_argument("-nf", "--nb_frames", help = "define board size", required = False, default = 4, type = int)
            self.parser.add_argument("-na", "--nb_actions", help = "define board size", required = False, default = 5, type = int)
            self.parser.add_argument("-uf", "--update_freq", help = "frequency to update target", required = False, default = 500, type = int)

            self.args = self.parser.parse_args()
            self.status_load = False
            self.status_visual = False
            self.local_state = False
            self.dueling = False
            self.double = False
            self.per = False

            if self.args.load:
                script_dir = path.dirname(__file__) # Absolute dir the script is in
                abs_file_path = path.join(script_dir, self.args.load)
                model = load_model(abs_file_path)

                self.status_load = True

            if self.args.visual:
                self.status_visual = True

            if self.args.local_state:
                self.local_state = True

            if self.args.dueling:
                self.dueling = True

            if self.args.double:
                self.double = True

            if self.args.per:
                self.per = True

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
