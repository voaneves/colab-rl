#!/usr/bin/env python

""" Needs update!
"""

from os import path

from tensorflow.keras.models import load_model
import tensorflow as tf
from keras.optimizers import *

from game.snake.snake import Game
from models.dqn import Agent
from models.utilities.noisy_dense import NoisyDenseFG
from models.utilities.networks import create_model
from models.utilities.misc import HandleArguments

VISUAL = True
RELATIVE_POS = False
TIMEIT_TRAIN = False
TIMEIT_TEST = False
BOARD_SIZE = 10
LOCAL_STATE = True
PER = False
BATCH_SIZE = 64
NB_FRAMES = 4
MEMORY_SIZE = -1
NB_EPOCH_TEST = 100
GAMMA = 0.95
UPDATE_TARGET_FREQ = 500
NB_FRAMES = 4
BOARD_SIZE = 10
LOSS = 'huber_loss'
DUELING = True

def main():
    script_dir = path.dirname(__file__) # Absolute dir the script is in
    abs_file_path = path.join(script_dir, 'models/keras_model.h5')
    # Python 3.6
    #model = load_model(abs_file_path, custom_objects = {'clipped_error': clipped_error,
    #                                                    'NoisyDenseFG': NoisyDenseFG})

    # Python 3.7

    game = Game(player = "ROBOT",
                board_size = BOARD_SIZE,
                local_state = LOCAL_STATE,
                relative_pos = RELATIVE_POS)

    with tf.Session() as sess:
        model = create_model(optimizer = RMSprop(),
                             loss = 'huber_loss',
                             stack = NB_FRAMES,
                             input_size = BOARD_SIZE,
                             output_size = game.nb_actions,
                             dueling = DUELING)
        sess.run(tf.global_variables_initializer())
        function = load_model(abs_file_path,
                              custom_objects = {'huber_loss': tf.losses.huber_loss,
                                                'NoisyDenseFG': NoisyDenseFG,
                                                'NoisyDenseIG': NoisyDenseIG,
                                                'tf': tf})
        model.set_weights(function.get_weights())
        target = None

        agent = Agent(model = model,
                      target = target,
                      sess = sess,
                      memory_size = MEMORY_SIZE,
                      nb_frames = NB_FRAMES,
                      board_size = BOARD_SIZE,
                      per = PER,
                      update_target_freq = UPDATE_TARGET_FREQ)
        agent.test(game,
                   nb_epoch = NB_EPOCH_TEST,
                   visual = VISUAL)

if __name__ == '__main__':
    main()
