#!/usr/bin/env python

"""dqn: First try to create an AI for SnakeGame. Is it good enough?

This algorithm is a implementation of DQN, Double DQN logic (using a target
network to have fixed Q-targets), Dueling DQN logic (Q(s,a) = Advantage + Value)
and PER (Prioritized Experience Replay, using Sum Trees). You can read more
about these on https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682

Possible usage:
    * Simple DQN;
    * DDQN;
    * DDDQN;
    * DDDQN + PER;
    * a combination of any of the above.

Arguments:
    --load FILE.h5: load a previously trained model in '.h5' format.
    --board_size INT: assign the size of the board, default = 10
    --nb_frames INT: assign the number of frames per stack, default = 4.
    --nb_actions INT: assign the number of actions possible, default = 5.
    --update_freq INT: assign how often, in epochs, to update the target,
      default = 500.
    --visual: select wheter or not to draw the game in pygame.
    --double: use a target network with double DQN logic.
    --dueling: use dueling network logic, Q(s,a) = A + V.
    --per: use Prioritized Experience Replay (based on Sum Trees).
    --local_state: Verify is possible next moves are dangerous (field expertise)
"""

import numpy as np
from os import path, environ, sys
import random

import inspect # Making relative imports from parallel folders possible
currentdir = path.dirname(path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = path.dirname(currentdir)
sys.path.insert(0, parentdir)

from keras.optimizers import RMSprop, Nadam
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')

from game.snake import Game
from utilities.networks import CNN1, CNN2, CNN3, CNN_DUELING
from utilities.clipped_error import clipped_error
from utilities.argument_handler import HandleArguments
from utilities.policy import *
from memory import ExperienceReplay

__author__ = "Victor Neves"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Victor Neves"
__email__ = "victorneves478@gmail.com"
__status__ = "Production"

class Agent:
    """Agent based in a simple DQN that can read states, remember and play.

    Attributes:
    memory: memory used in the model. Input memory or ExperienceReplay.
    model: the input model, Conv2D in Keras.
    target: the target model, used to calculade the fixed Q-targets.
    nb_frames: ammount of frames for each sars.
    frames: the frames in each sars.
    per: flag for PER usage.
    """
    def __init__(self, model, target, memory = None, memory_size = 150000,
                 nb_frames = 4, board_size = 10, per = False):
        """Initialize the agent with given attributes."""
        if memory:
            self.memory = memory
        else:
            self.memory = ExperienceReplay(memory_size = memory_size, per = per)

        self.per = per
        self.model = model
        self.target = target
        self.nb_frames = nb_frames
        self.board_size = board_size
        self.frames = None
        self.target_updates = 0

    def reset_memory(self):
        """Reset memory if necessary."""
        self.memory.reset_memory()

    def get_game_data(self, game):
        """Create a list with 4 frames and append/pop them each frame."""
        if game.game_over:
            frame = np.zeros((self.board_size, self.board_size))
        else:
            frame = game.state()

        if self.frames is None:
            self.frames = [frame] * self.nb_frames
        else:
            self.frames.append(frame)
            self.frames.pop(0)

        return np.expand_dims(self.frames, 0)

    def clear_frames(self):
        """Reset frames to restart appending."""
        self.frames = None

    def update_target_model(self):
        """Update the target model with the main model's weights."""
        self.target_updates += 1
        self.target.set_weights(self.model.get_weights())

    def print_metrics(self, epoch, nb_epoch, history_size, history_loss,
                      history_step, history_reward, policy, value, win_count,
                      verbose = 1):
        """"""
        if verbose == 0:
            pass
        elif verbose == 1:
            print("Epoch: {:03d}/{:03d} | Size: {:03d} | Ammount of steps: {:03d} | Wins: {:d} | Win percentage: {:.1f}%".format(epoch + 1, nb_epoch, history_size[-1], history_step[-1], win_count, 100 * win_count/(epoch + 1)))
        else:
            # Print epoch info:
            print("Epoch: {:03d}/{:03d}".format(epoch + 1, nb_epoch))

            # Print training performance:
            if epoch > 100:
                print('\t\x1b[0;30;47m' + ' Training metrics ' + '\x1b[0m'
                      + '\tTotal loss: {:.4f} | Loss per step: {:.4f} | Mean loss - 100 episodes: {:.4f}'.format(history_loss[-1], history_loss[-1]/history_step[-1], sum(history_loss[-100:]) / 100))
                print('\t\x1b[0;30;47m' + ' Game metrics ' + '\x1b[0m'
                      + "\t\tSize: {:d} | Ammount of steps: {:d} | Steps per food eaten: {:.1f} | Mean size - 100 episodes: {:.1f}".format(history_size[-1], history_step[-1], history_size[-1] / history_step[-1], sum(history_step[-100:]) / 100))
            else:
                print('\t\x1b[0;30;47m' + ' Training metrics ' + '\x1b[0m'
                      + "\tTotal loss: {:.4f} | Loss per step: {:.4f}".format(history_loss[-1], history_loss[-1]/history_step[-1]))
                print('\t\x1b[0;30;47m' + ' Game metrics ' + '\x1b[0m' +
                      "\t\tSize: {:d} | Ammount of steps: {:d} | Steps per food eaten: {:.1f}".format(history_size[-1], history_step[-1], history_size[-1] / history_step[-1]))

            # Print policy metrics
            if policy == "BoltzmannQPolicy":
                print('\t\x1b[0;30;47m' + ' Policy metrics ' + '\x1b[0m'
                      + "\tBoltzmann Temperature: {:.2f} | Episode reward: {:.1f} | Wins: {:d} | Win percentage: {:.1f}%".format(value, history_reward[-1], win_count, 100 * win_count/(epoch + 1)))
            elif policy == "BoltzmannGumbelQPolicy":
                print('\t\x1b[0;30;47m' + ' Policy metrics ' + '\x1b[0m'
                      + "\tNumber of actions: {:.0f} | Episode reward: {:.1f} | Wins: {:d} | Win percentage: {:.1f}%".format(value, history_reward[-1], win_count, 100 * win_count/(epoch + 1)))
            else:
                print('\t\x1b[0;30;47m' + ' Policy metrics ' + '\x1b[0m'
                      + "\tEpsilon: {:.2f} | Episode reward: {:.1f} | Wins: {:d} | Win percentage: {:.1f}%".format(value, history_reward[-1], win_count, 100 * win_count/(epoch + 1)))

    def train(self, game, nb_epoch = 10000, batch_size = 64, gamma = 0.95,
              eps = [1., .01], temp = [1., 0.01], learning_rate = 0.5,
              observe = 0, update_target_freq = 500, rounds = 1,
              policy = "EpsGreedyQPolicy", verbose = 1):
        """The main training function, loops the game, remember and choose best
        action given game state (frames)."""

        history_size = []
        history_step = []
        history_loss = []
        history_reward = []

        if policy == "BoltzmannQPolicy":
            q_policy = BoltzmannQPolicy(temp[0], temp[1], nb_epoch * learning_rate)
        if policy == "BoltzmannGumbelQPolicy":
            q_policy = BoltzmannGumbelQPolicy()
        else:
            q_policy = EpsGreedyQPolicy(eps[0], eps[1], nb_epoch * learning_rate)

        nb_actions = game.nb_actions
        win_count = 0
        for turn in range(optim_rounds):
            if turn > 0:
                for epoch in range(nb_epoch):
                    if epoch >= observe: # Get the batchs and train
                        batch = self.memory.get_targets(model = self.model,
                                                        target = self.target,
                                                        batch_size = batch_size,
                                                        gamma = gamma,
                                                        nb_actions = nb_actions)

                        if batch:
                            inputs, targets, IS_weights = batch

                            if inputs is not None and targets is not None:
                                loss = float(self.model.train_on_batch(inputs,
                                                                       targets,
                                                                       IS_weights))
                                print("Optimizer turn: {:2d} | Epoch: {:03d}/{:03d} | Loss: {:.4f}".format(turn, epoch + 1, nb_epoch, loss))
            else:
                for epoch in range(nb_epoch):
                    loss = 0.
                    total_reward = 0.
                    game.reset()
                    self.clear_frames()

                    S = self.get_game_data(game)

                    while not game.game_over:
                        game.food_pos = game.generate_food()
                        action, value = q_policy.select_action(self.model, S, epoch, nb_actions)

                        game.play(action, "ROBOT")
                        r = game.get_reward()
                        total_reward += r

                        S_prime = self.get_game_data(game)
                        experience = [S, action, r, S_prime, game.game_over]
                        self.memory.remember(*experience) # Add to the memory
                        S = S_prime # Advance to the next state (stack of S)

                        if epoch >= observe: # Get the batchs and train
                            batch = self.memory.get_targets(model = self.model,
                                                            target = self.target,
                                                            batch_size = batch_size,
                                                            gamma = gamma,
                                                            nb_actions = nb_actions)

                            if batch:
                                inputs, targets, IS_weights = batch

                                if inputs is not None and targets is not None:
                                    loss += float(self.model.train_on_batch(inputs,
                                                                            targets,
                                                                            IS_weights))

                    if game.is_won():
                        win_count += 1 # Counter for metric purposes

                    if self.per: # Advance beta
                        self.memory.per_beta = self.memory.schedule.value(epoch)

                    if self.target is not None: # Update the target model
                        if epoch % update_target_freq == 0:
                            self.update_target_model()

                    history_size.append(game.snake.length)
                    history_step.append(game.step)
                    history_loss.append(loss)
                    history_reward.append(total_reward)

                    self.print_metrics(epoch, nb_epoch, history_size, history_loss,
                                       history_step, history_reward, policy, value,
                                       win_count, verbose)

    def play(self, game, nb_epoch = 1000, eps = 0.01, temp = 0.01,
             visual = False, policy = "GreedyQPolicy"):
        """Play the game with the trained agent. Can use the visual tag to draw
            in pygame."""
        win_count = 0
        result_size = []
        result_step = []
        if policy == "BoltzmannQPolicy":
            q_policy = BoltzmannQPolicy(temp, temp, nb_epoch)
        elif policy == "EpsGreedyQPolicy":
            q_policy = EpsGreedyQPolicy(eps, eps, nb_epoch)
        else:
            q_policy = GreedyQPolicy()

        for epoch in range(nb_epoch):
            game.reset()
            self.clear_frames()
            S = self.get_game_data(game)

            if visual:
                game.create_window()
                # The main loop, it pump key_presses and update every tick.
                environ['SDL_VIDEO_CENTERED'] = '1' # Centering the window
                previous_size = game.snake.length # Initial size of the snake
                color_list = game.gradient([(42, 42, 42), (152, 152, 152)],\
                                               previous_size)

            while not game.game_over:
                action, value = q_policy.select_action(self.model, S, epoch, nb_actions)
                game.play(action, "ROBOT")
                current_size = game.snake.length # Update the body size

                if visual:
                    game.draw(color_list)

                    if current_size > previous_size:
                        color_list = game.gradient([(42, 42, 42), (152, 152, 152)],
                                                   game.snake.length)

                        previous_size = current_size

                S = self.get_game_data(game)

                if game.game_over:
                    result_size.append(current_size)
                    result_step.append(game.step)

            if game.is_won():
                win_count += 1

        print("Accuracy: {} %".format(100. * win_count / nb_epoch))
        print("Mean size: {} | Biggest size: {} | Smallest size: {}"\
              .format(np.mean(result_size), np.max(result_size),
                      np.min(result_size)))
        print("Mean steps: {} | Biggest step: {} | Smallest step: {}"\
              .format(np.mean(result_step), np.max(result_step),\
                      np.min(result_step)))

if __name__ == '__main__':
    arguments = HandleArguments()
    board_size = arguments.args.board_size
    nb_actions = arguments.args.nb_actions
    nb_frames = arguments.args.nb_frames
    update_target_freq = arguments.args.update_freq

    if not arguments.status_visual:
        if not arguments.status_load:
            if arguments.dueling:
                model = CNN_DUELING(optimizer = RMSprop(), loss = clipped_error,
                                    stack = nb_frames, input_size = board_size,
                                    output_size = nb_actions)

                target = None
                if arguments.double:
                    target = CNN_DUELING(optimizer = RMSprop(),
                                         loss = clipped_error,
                                         stack = nb_frames,
                                         input_size = board_size,
                                         output_size = nb_actions)

            else:
                model = CNN1(optimizer = RMSprop(), loss = clipped_error,
                            stack = nb_frames, input_size = board_size,
                            output_size = nb_actions)

                target = None
                if arguments.double:
                    target = CNN1(optimizer = RMSprop(), loss = clipped_error,
                                  stack = nb_frames, input_size = board_size,
                                  output_size = nb_actions)

            print("Not using --load. Default behavior is to train the model "
                  + "and then play. Training:")

            game = Game(board_size = board_size,
                        local_state = arguments.local_state, relative_pos = False)
            agent = Agent(model = model, target = target, memory_size = -1,
                          nb_frames = nb_frames, board_size = board_size,
                          per = arguments.per)
            agent.train(game, batch_size = 64, nb_epoch = 10000, gamma = 0.8,
                        update_target_freq = update_target_freq)
        else:
            game = Game(board_size = board_size,
                        local_state = arguments.local_state, relative_pos = False)
            agent = Agent(model = model, target = target, memory_size = -1,
                          nb_frames = nb_frames, board_size = board_size,
                          per = arguments.per)

            print("Loading file located in {}. We can play after that."
                    .format(arguments.args.load))

        print("--visual not used. Default behavior is to have drawing disabled."
              + " Playing:")
        agent.play(game, visual = False)
    else:
        if not arguments.status_load:
            if arguments.dueling:
                model = CNN_DUELING(optimizer = RMSprop(), loss = clipped_error,
                                    stack = nb_frames, input_size = board_size,
                                    output_size = nb_actions)
                target = None
                if arguments.double:
                    target = CNN_DUELING(optimizer = RMSprop(),
                                         loss = clipped_error,
                                         stack = nb_frames,
                                         input_size = board_size,
                                         output_size = nb_actions)

            else:
                model = CNN1(optimizer = RMSprop(), loss = clipped_error,
                            stack = nb_frames, input_size = board_size,
                            output_size = nb_actions)

                target = None
                if arguments.double:
                    target = CNN1(optimizer = RMSprop(), loss = clipped_error,
                                  stack = nb_frames, input_size = board_size,
                                  output_size = nb_actions)

            print("Not using --load. Default behavior is to train the model and"
                  + "then play. Training:")

            game = Game(board_size = board_size,
                        local_state = arguments.local_state, relative_pos = False)
            agent = Agent(model = model, target = target, memory_size = -1,
                          nb_frames = nb_frames, board_size = board_size,
                          per = arguments.per)
            agent.train(game, batch_size = 64, nb_epoch = 10000, gamma = 0.8,
                        update_target_freq = update_target_freq)
        else:
            game = Game(board_size = board_size,
                        local_state = arguments.local_state, relative_pos = False)
            agent = Agent(model = model, target = target, memory_size = -1,
                          nb_frames = nb_frames, board_size = board_size,
                          per = arguments.per)

            print("Loading file located in {}. We can play after that."\
                  .format(arguments.args.load))

        print("--visual is activated. Drawing the board and controlled by the"
              + "DQN Agent. Playing:")
        agent.play(game, visual = True)
