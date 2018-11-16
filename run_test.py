#!/usr/bin/env python

""" Needs update!
"""

from os import path

from keras.models import load_model

from game.snake import Game
from models.dqn import Agent
from models.utilities.noisy_dense import NoisyDenseFG
from models.utilities.networks import create_model, create_noisy_model
from models.utilities.misc import HandleArguments, clipped_error

VISUAL = False
RELATIVE_POS = False
TIMEIT_TRAIN = False
TIMEIT_TEST = False
BOARD_SIZE = 10
LOCAL_STATE = True
PER = False
BATCH_SIZE = 64
NB_FRAMES = 4
MEMORY_SIZE = -1
NB_EPOCH_TEST = 1000
GAMMA = 0.95
UPDATE_TARGET_FREQ = 500

def main():
    script_dir = path.dirname(__file__) # Absolute dir the script is in
    abs_file_path = path.join(script_dir, 'models/keras.h5')
    # Python 3.6
    #model = load_model(abs_file_path, custom_objects = {'clipped_error': clipped_error,
    #                                                    'NoisyDenseFG': NoisyDenseFG})

    # Python 3.7
    model = load_model(abs_file_path, custom_objects = {'clipped_error': clipped_error,
                                                        'NoisyDenseFG': NoisyDenseFG})
    target = None

    game = Game(player = "ROBOT", board_size = BOARD_SIZE,
                local_state = LOCAL_STATE, relative_pos = RELATIVE_POS)
    agent = Agent(model = model, target = target, memory_size = MEMORY_SIZE,
                  nb_frames = NB_FRAMES, board_size = BOARD_SIZE,
                  per = PER, update_target_freq = UPDATE_TARGET_FREQ)
    agent.test(game, nb_epoch = NB_EPOCH_TEST, visual = VISUAL)

if __name__ == '__main__':
    main()
