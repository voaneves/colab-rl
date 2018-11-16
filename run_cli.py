#!/usr/bin/env python

""" Needs update!
"""

from os import path

import tensorflow as tf
from keras.models import load_model

from game.snake import Game
from models.dqn import Agent
from models.utilities.networks import create_model, create_noisy_model
from models.utilities.misc import HandleArguments

RELATIVE_POS = False
TIMEIT_TRAIN = False
TIMEIT_TEST = False
BATCH_SIZE = 64
NB_EPOCH = 10000
NB_EPOCH_TEST = 1000
GAMMA = 0.95

def main():
    arguments = HandleArguments()

    load = arguments.args.load
    visual = arguments.args.visual
    cnn = arguments.args.cnn_model
    optimizer = arguments.args.optimizer
    error = arguments.args.error
    local_state = arguments.args.local_state
    double = arguments.args.double
    dueling = arguments.args.dueling
    per = arguments.args.per
    noisy = arguments.args.noisy_net
    benchmark = arguments.args.benchmark
    n_steps = arguments.args.n_steps
    board_size = arguments.args.board_size
    memory_size = arguments.args.memory_size
    nb_frames = arguments.args.nb_frames
    update_target_freq = arguments.args.update_freq

    if not load:
        print("Not using -load. Default behavior is to train the model "
              + "and then play. Training:")

        game = Game(player = "ROBOT", board_size = board_size,
                    local_state = local_state, relative_pos = RELATIVE_POS)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            model = create_model(optimizer = optimizer, loss = error,
                                 stack = nb_frames, input_size = board_size,
                                 output_size = game.nb_actions,
                                 dueling = dueling, cnn = cnn, noisy = noisy)
            target = None

            if double:
                target = create_model(optimizer = optimizer, loss = error,
                                      stack = nb_frames, input_size = board_size,
                                      output_size = game.nb_actions,
                                      dueling = dueling, cnn = cnn, noisy = noisy)

            agent = Agent(model = model, target = target, memory_size = memory_size,
                          nb_frames = nb_frames, board_size = board_size,
                          per = per, update_target_freq = update_target_freq)

            agent.train(game, batch_size = BATCH_SIZE, nb_epoch = NB_EPOCH,
                        gamma = GAMMA, n_steps = n_steps)
    else:
        game = Game(player = "ROBOT", board_size = board_size,
                    local_state = local_state, relative_pos = RELATIVE_POS)

        script_dir = path.dirname(__file__) # Absolute dir the script is in
        abs_file_path = path.join(script_dir, self.args.load)
        model = load_model(abs_file_path)
        
        agent = Agent(model = model, target = target, memory_size = memory_size,
                      nb_frames = nb_frames, board_size = board_size,
                      per = per, update_target_freq = update_target_freq)

        print("Loading file located in {}. We can play after that."
                .format(arguments.args.load))

    if benchmark:
        print("--benchmark is activated. Playing for NB_EPOCH_TEST episodes.")

        agent.test(game, nb_epoch = NB_EPOCH_TEST, visual = visual)

if __name__ == '__main__':
    main()
