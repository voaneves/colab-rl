#!/usr/bin/env python

"""dqn: First try to create an AI for SnakeGame. Is it good enough?

This algorithm is a implementation of DQN, Double DQN logic (using a target
network to have fixed Q-targets), Dueling DQN logic (Q(s,a) = Advantage + Value),
PER (Prioritized Experience Replay, using Sum Trees) and Multi-step returns. You
can read more about these on https://goo.gl/MctLzp

Implemented algorithms
----------
    * Simple Deep Q-network (DQN with ExperienceReplay);
        Paper: https://arxiv.org/abs/1312.5602
    * Double Deep Q-network (Double DQN);
        Paper: https://arxiv.org/abs/1509.06461
    * Dueling Deep Q-network (Dueling DQN);
        Paper: https://arxiv.org/abs/1511.06581
    * Prioritized Experience Replay (PER);
        Paper: https://arxiv.org/abs/1511.05952
    * Multi-step returns (n-steps);
        Paper: https://arxiv.org/pdf/1703.01327
    * Noisy nets.
        Paper: https://arxiv.org/abs/1706.10295

Arguments
----------
--load: 'file.h5'
    Load a previously trained model in '.h5' format.
--board_size: int, optional, default = 10
    Assign the size of the board.
--nb_frames: int, optional, default = 4
    Assign the number of frames per stack, default = 4.
--nb_actions: int, optional, default = 5
    Assign the number of actions possible.
--update_freq: int, optional, default = 0.001
    Whether to soft or hard update the target. Epochs or ammount of the update.
--visual: boolean, optional, default = False
    Select wheter or not to draw the game in pygame.
--double: boolean, optional, default = False
    Use a target network with double DQN logic.
--dueling: boolean, optional, default = False
    Whether to use dueling network logic, Q(s,a) = A + V.
--per: boolean, optional, default = False
    Use Prioritized Experience Replay (based on Sum Trees).
--local_state: boolean, optional, default = True
    Verify is possible next moves are dangerous (field expertise)
"""

import numpy as np
from array import array
import random

import pygame
from .utilities.policy import GreedyQPolicy, EpsGreedyQPolicy, BoltzmannQPolicy,\
                             BoltzmannGumbelQPolicy
from .memory import ExperienceReplay, PrioritizedExperienceReplay

__author__ = "Victor Neves"
__license__ = "MIT"
__maintainer__ = "Victor Neves"
__email__ = "victorneves478@gmail.com"
__status__ = "Production"


class Agent:
    """Agent based in a simple DQN that can read states, remember and play.

    Attributes
    ----------
    memory: object
        Memory used in training. ExperienceReplay or PrioritizedExperienceReplay
    memory_size: int, optional, default = -1
        Capacity of the memory used.
    model: keras model
        The input model in Keras.
    target: keras model, optional, default = None
        The target model, used to calculade the fixed Q-targets.
    nb_frames: int, optional, default = 4
        Ammount of frames for each experience (sars).
    board_size: int, optional, default = 10
        Size of the board used.
    frames: list of experiences
        The buffer of frames, store sars experiences.
    per: boolean, optional, default = False
        Flag for PER usage.
    update_target_freq: int or float, default = 0.001
        Whether soft or hard updates occur. If < 1, soft updated target model.
    n_steps: int, optional, default = 1
        Size of the rewards buffer, to use Multi-step returns.
    """
    def __init__(self, model, target = None, memory_size = -1, nb_frames = 4,
                 board_size = 10, per = False, update_target_freq = 0.001):
        """Initialize the agent with given attributes."""
        if per:
            self.memory = PrioritizedExperienceReplay(memory_size = memory_size)
        else:
            self.memory = ExperienceReplay(memory_size = memory_size)

        self.per = per
        self.model = model
        self.target = target
        self.nb_frames = nb_frames
        self.board_size = board_size
        self.update_target_freq = update_target_freq
        self.set_noise_list()
        self.clear_frames()

    def reset_memory(self):
        """Reset memory if necessary."""
        self.memory.reset_memory()

    def set_noise_list(self):
        """Set a list of noise variables if NoisyNet is involved."""
        self.noise_list = []
        for layer in self.model.layers:
            if type(layer) in {NoisyDenseFG}:
                self.noise_list.extend(layer.noise_list)

    def sample_noise(self):
        """Resample noise variables in NoisyNet."""
        for noise in self.noise_list:
            self.sess.run(noise.initializer)

    def get_game_data(self, game):
        """Create a list with 4 frames and append/pop them each frame.

        Return
        ----------
        expanded_frames: list of experiences
            The buffer of frames, shape = (nb_frames, board_size, board_size)
        """
        frame = game.state()

        if self.frames is None:
            self.frames = [frame] * self.nb_frames
        else:
            self.frames.append(frame)
            self.frames.pop(0)

        expanded_frames = np.expand_dims(self.frames, 0)
        expanded_frames = np.transpose(expanded_frames, [0, 3, 2, 1])

        return expanded_frames

    def clear_frames(self):
        """Reset frames to restart appending."""
        self.frames = None

    def update_target_model_hard(self):
        """Update the target model with the main model's weights."""
        self.target.set_weights(self.model.get_weights())

    def transfer_weights(self):
        """Transfer Weights from Model to Target at rate update_target_freq."""
        model_weights = self.model.get_weights()
        target_weights = self.target.get_weights()

        for i in range(len(W)):
            target_weights[i] = (self.update_target_freq * model_weights[i]
                                 + ((1 - self.update_target_frequency)
                                    * target_weights[i]))

        self.target.set_weights(target_weights)

    def print_metrics(self, epoch, nb_epoch, history_size, policy, value,
                      win_count, history_step, history_reward,
                      history_loss = None, verbose = 1):
        """Function to print metrics of training steps."""
        if verbose == 0:
            pass
        elif verbose == 1:
            text_epoch = ('Epoch: {:03d}/{:03d} | Mean size 10: {:.1f} | '
                           + 'Longest 10: {:03d} | Mean steps 10: {:.1f} | '
                           + 'Wins: {:d} | Win percentage: {:.1f}%')
            print(text_epoch.format(epoch + 1, nb_epoch,
                                    np.mean(history_size[-10:]),
                                    max(history_size[-10:]),
                                    np.mean(history_step[-10:]),
                                    win_count, 100 * win_count/(epoch + 1)))
        else:
            text_epoch = 'Epoch: {:03d}/{:03d}'  # Print epoch info
            print(text_epoch.format(epoch + 1, nb_epoch))

            if loss is not None:  # Print training performance
                text_train = ('\t\x1b[0;30;47m' + ' Training metrics ' + '\x1b[0m'
                              + '\tTotal loss: {:.4f} | Loss per step: {:.4f} | '
                              + 'Mean loss - 100 episodes: {:.4f}')
                print(text_perf.format(history_loss[-1],
                                       history_loss[-1] / history_step[-1],
                                       np.mean(history_loss[-100:])))

            text_game = ('\t\x1b[0;30;47m' + ' Game metrics ' + '\x1b[0m'
                         + '\t\tSize: {:d} | Ammount of steps: {:d} | '
                         + 'Steps per food eaten: {:.1f} | '
                         + 'Mean size - 100 episodes: {:.1f}')
            print(text_game.format(history_size[-1], history_step[-1],
                                   history_size[-1] / history_step[-1],
                                   np.mean(history_step[-100:])))

            # Print policy metrics
            if policy == "BoltzmannQPolicy":
                text_policy = ('\t\x1b[0;30;47m' + ' Policy metrics ' + '\x1b[0m'
                               + '\tBoltzmann Temperature: {:.2f} | '
                               + 'Episode reward: {:.1f} | Wins: {:d} | '
                               + 'Win percentage: {:.1f}%')
                print(text_policy.format(value, history_reward[-1], win_count,
                                         100 * win_count/(epoch + 1)))
            elif policy == "BoltzmannGumbelQPolicy":
                text_policy = ('\t\x1b[0;30;47m' + ' Policy metrics ' + '\x1b[0m'
                               + '\tNumber of actions: {:.0f} | '
                               + 'Episode reward: {:.1f} | Wins: {:d} | '
                               + 'Win percentage: {:.1f}%')
                print(text_policy.format(value, history_reward[-1], win_count,
                                         100 * win_count/(epoch + 1)))
            else:
                text_policy = ('\t\x1b[0;30;47m' + ' Policy metrics ' + '\x1b[0m'
                               + '\tEpsilon: {:.2f} | Episode reward: {:.1f} | '
                               + 'Wins: {:d} | Win percentage: {:.1f}%')
                print(text_policy.format(value, history_reward[-1], win_count,
                                         100 * win_count/(epoch + 1)))

    def train_model(self, model, target, batch_size, gamma, nb_actions, epoch = 0):
        """Function to train the model on a batch of the data. The optimization
        flag is used when we are not playing, just batching and optimizing.

        Return
        ----------
        loss: float
            Training loss of given batch.
        """
        loss = 0.
        batch = self.memory.get_targets(model = self.model,
                                        target = self.target,
                                        batch_size = batch_size,
                                        gamma = gamma,
                                        nb_actions = nb_actions,
                                        n_steps = self.n_steps)

        if batch:
            inputs, targets, IS_weights = batch

            if inputs is not None and targets is not None:
                loss = float(self.model.train_on_batch(inputs,
                                                       targets,
                                                       IS_weights))

        return loss

    def train(self, game, nb_epoch = 10000, batch_size = 64, gamma = 0.95,
              eps = [1., .01], temp = [1., 0.01], learning_rate = 0.5,
              observe = 0, optim_rounds = 1, policy = "EpsGreedyQPolicy",
              verbose = 1, n_steps = 1):
        """The main training function, loops the game, remember and choose best
        action given game state (frames)."""
        if not hasattr(self, 'n_steps'):
            self.n_steps = n_steps  # Set attribute only once

        history_size = array('i')  # Holds all the sizes
        history_step = array('f')  # Holds all the steps
        history_loss = array('f')  # Holds all the losses
        history_reward = array('f')  # Holds all the rewards

        # Select exploration policy. EpsGreedyQPolicy runs faster, but takes
        # longer to converge. BoltzmannGumbelQPolicy is the slowest, but
        # converge really fast (0.1 * nb_epoch used in EpsGreedyQPolicy).
        # BoltzmannQPolicy is in the middle.
        if policy == "BoltzmannQPolicy":
            q_policy = BoltzmannQPolicy(temp[0], temp[1], nb_epoch * learning_rate)
        elif policy == "BoltzmannGumbelQPolicy":
            q_policy = BoltzmannGumbelQPolicy()
        else:
            q_policy = EpsGreedyQPolicy(eps[0], eps[1], nb_epoch * learning_rate)

        nb_actions = game.nb_actions
        win_count = 0

        # If optim_rounds is bigger than one, the model will keep optimizing
        # after the exploration, in turns of nb_epoch size.
        for turn in range(optim_rounds):
            if turn > 0:
                for epoch in range(nb_epoch):
                    loss = self.train_model(model = self.model,
                                            epoch = epoch,
                                            target = self.target,
                                            batch_size = batch_size,
                                            gamma = gamma,
                                            nb_actions = nb_actions)
                    text_optim = ('Optimizer turn: {:2d} | Epoch: {:03d}/{:03d}'
                                  + '| Loss: {:.4f}')
                    print(text_optim.format(turn, epoch + 1, nb_epoch, loss))
            else:  # Exploration and training
                for epoch in range(nb_epoch):
                    loss = 0.
                    total_reward = 0.
                    game.reset_game()
                    self.clear_frames()
                    S = self.get_game_data(game)

                    if n_steps > 1:  # Create multi-step returns buffer.
                        n_step_buffer = array('f')

                    while not game.game_over:  # Main loop, until game_over
                        game.food_pos = game.generate_food()
                        self.sample_noise()
                        action, value = q_policy.select_action(self.model,
                                                               S, epoch,
                                                               nb_actions)
                        game.play(action)
                        r = game.get_reward()
                        total_reward += r

                        if n_steps > 1:
                            n_step_buffer.append(r)

                            if len(n_step_buffer) < n_steps:
                                R = r
                            else:
                                R = sum([n_step_buffer[i] * (gamma ** i)\
                                        for i in range(n_steps)])

                                n_step_buffer.pop(0)
                        else:
                            R = r

                        S_prime = self.get_game_data(game)
                        experience = [S, action, R, S_prime, game.game_over]
                        self.memory.remember(*experience)  # Add to the memory
                        S = S_prime  # Advance to the next state (stack of S)

                        if epoch >= observe:  # Get the batchs and train
                            loss += self.train_model(model = self.model,
                                                     target = self.target,
                                                     batch_size = batch_size,
                                                     gamma = gamma,
                                                     nb_actions = nb_actions)

                    if game.is_won():
                        win_count += 1  # Counter of wins for metrics

                    if self.per:  # Advance beta, used in PER
                        self.memory.beta = self.memory.schedule.value(epoch)

                    if self.target is not None:  # Update the target model
                        if update_target_freq >= 1: # Hard updates
                            if epoch % self.update_target_freq == 0:
                                self.update_target_model_hard()
                        elif update_target_freq < 1.:  # Soft updates
                            self.transfer_weights()

                    history_size.append(game.snake.length)
                    history_step.append(game.step)
                    history_loss.append(loss)
                    history_reward.append(total_reward)

                    if (epoch + 1) % 10 == 0:
                        self.print_metrics(epoch = epoch, nb_epoch = nb_epoch,
                                           history_size = history_size,
                                           history_loss = history_loss,
                                           history_step = history_step,
                                           history_reward = history_reward,
                                           policy = policy, value = value,
                                           win_count = win_count,
                                           verbose = verbose)

    def test(self, game, nb_epoch = 1000, eps = 0.01, temp = 0.01,
             visual = False, policy = "GreedyQPolicy"):
        """Play the game with the trained agent. Can use the visual tag to draw
            in pygame."""
        win_count = 0

        history_size = array('i')  # Holds all the sizes
        history_step = array('f')  # Holds all the steps
        history_reward = array('f')  # Holds all the rewards

        if policy == "BoltzmannQPolicy":
            q_policy = BoltzmannQPolicy(temp, temp, nb_epoch)
        elif policy == "EpsGreedyQPolicy":
            q_policy = EpsGreedyQPolicy(eps, eps, nb_epoch)
        else:
            q_policy = GreedyQPolicy()

        for epoch in range(nb_epoch):
            game.reset_game()
            self.clear_frames()

            if visual:
                game.create_window()
                previous_size = game.snake.length  # Initial size of the snake
                color_list = game.gradient([(42, 42, 42), (152, 152, 152)],\
                                               previous_size)
                elapsed = 0

            while not game.game_over:
                if visual:
                    elapsed += game.fps.get_time()  # Get elapsed time since last call.

                    if elapsed >= 60:
                        elapsed = 0
                        S = self.get_game_data(game)
                        action, value = q_policy.select_action(self.model, S, epoch, game.nb_actions)
                        game.play(action)
                        current_size = game.snake.length  # Update the body size

                        if current_size > previous_size:
                            color_list = game.gradient([(42, 42, 42), (152, 152, 152)],
                                                       game.snake.length)

                            previous_size = current_size

                        game.draw(color_list)

                    pygame.display.update()
                    game.fps.tick(120)  # Limit FPS to 100
                else:
                    S = self.get_game_data(game)
                    action, value = q_policy.select_action(self.model, S, epoch, game.nb_actions)
                    game.play(action)
                    current_size = game.snake.length  # Update the body size

                if game.game_over:
                    history_size.append(current_size)
                    history_step.append(game.step)
                    history_reward.append(game.get_reward())

            if game.is_won():
                win_count += 1

        print("Accuracy: {} %".format(100. * win_count / nb_epoch))
        print("Mean size: {} | Biggest size: {} | Smallest size: {}"\
              .format(np.mean(history_size), np.max(history_size),
                      np.min(history_size)))
        print("Mean steps: {} | Biggest step: {} | Smallest step: {}"\
              .format(np.mean(history_step), np.max(history_step),
                      np.min(history_step)))
        print("Mean rewards: {} | Biggest reward: {} | Smallest reward: {}"\
              .format(np.mean(history_reward), np.max(history_reward),
                      np.min(history_reward)))
