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
from utilities.sum_tree import SumTree

__author__ = "Victor Neves"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Victor Neves"
__email__ = "victorneves478@gmail.com"
__status__ = "Production"

class ExperienceReplay:
    """The class that handles memory and experiences replay.

    Attributes:
        memory: memory array to insert experiences.
        memory_size: the ammount of experiences to be stored in the memory.
        input_shape: the shape of the input which will be stored.
        batch_function: returns targets according to S.
        per: flag for PER usage.
        per_epsilon: used to replace "0" probabilities cases.
        per_alpha: how much prioritization to use.
        per_beta: importance sampling weights (IS_weights).
    """
    def __init__(self, memory_size = 100, per = False, alpha = 0.6,
                 epsilon = 0.001, beta = 0.4, nb_epoch = 10000, decay = 0.5):
        """Initialize parameters and the memory array."""
        self.per = per
        self.memory_size = memory_size
        self.reset_memory() # Initiate the memory

        if self.per:
            self.per_epsilon = epsilon
            self.per_alpha = alpha
            self.per_beta = beta
            self.per_beta_inc = (1.0 - self.per_beta) / (nb_epoch * decay)

    def exp_size(self):
        """Returns how much memory is stored."""
        if self.per:
            return self.exp
        else:
            return len(self.memory)

    def get_priority(self, errors):
        """Returns priority based on how much prioritization to use."""
        return (errors + self.per_epsilon) ** self.per_alpha

    def update(self, tree_indices, errors):
        """Update a list of nodes, based on their errors."""
        priorities = self.get_priority(errors)

        for index, priority in zip(tree_indices, priorities):
            self.memory.update(index, priority)

    def remember(self, s, a, r, s_prime, game_over):
        """Remember SARS' experiences, with the game_over parameter (done)."""
        if not hasattr(self, 'input_shape'):
            self.input_shape = s.shape[1:] # set attribute only once

        experience = np.concatenate([s.flatten(),
                                     np.array(a).flatten(),
                                     np.array(r).flatten(),
                                     s_prime.flatten(),
                                     1 * np.array(game_over).flatten()])

        if self.per: # If using PER, insert in the max_priority.
            max_priority = self.memory.max_leaf()

            if max_priority == 0:
                max_priority = self.get_priority(0)

            self.memory.insert(experience, max_priority)
            self.exp += 1
        else: # Else, just append the experience to the list.
            self.memory.append(experience)

            if self.memory_size > 0 and self.exp_size() > self.memory_size:
                self.memory.pop(0)

    def get_samples(self, batch_size):
        """Sample the memory according to PER flag."""
        if self.per:
            batch = [None] * batch_size
            IS_weights = np.zeros((batch_size, ))
            tree_indices = np.zeros((batch_size, ), dtype = np.int32)

            memory_sum = self.memory.sum()
            len_seg = memory_sum / batch_size
            min_prob = self.memory.min_leaf() / memory_sum

            for i in range(batch_size):
                val = random.uniform(len_seg * i, len_seg * (i + 1))
                tree_indices[i], priority, batch[i] = self.memory.retrieve(val)
                prob = priority / self.memory.sum()
                IS_weights[i] = np.power(prob / min_prob, -self.per_beta)

            return np.array(batch), IS_weights, tree_indices

        else:
            return np.array(random.sample(self.memory, batch_size)), None, None

    def get_targets(self, target, model, batch_size, gamma = 0.9):
        """Function to sample, set batch function and use it for targets."""
        if self.exp_size() < batch_size:
            return None

        samples, IS_weights, tree_indices = self.get_samples(batch_size)
        nb_actions = model.get_output_shape_at(0)[-1] # Get number of actions
        input_dim = np.prod(self.input_shape) # Get the input shape, multiplied

        S = samples[:, 0 : input_dim] # Seperate the states
        a = samples[:, input_dim] # Separate the actions
        r = samples[:, input_dim + 1] # Separate the rewards
        S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2] # Next_actions
        game_over = samples[:, 2 * input_dim + 2] # Separate terminal flags

        # Reshape the arrays to make them usable by the model.
        r = r.repeat(nb_actions).reshape((batch_size, nb_actions))
        game_over = game_over.repeat(nb_actions)\
                             .reshape((batch_size, nb_actions))
        S = S.reshape((batch_size, ) + self.input_shape)
        S_prime = S_prime.reshape((batch_size, ) + self.input_shape)

        X = np.concatenate([S, S_prime], axis = 0)
        Y = model.predict(X)

        if target is not None: # Use Double DQN logic:
            actions = np.argmax(Y[batch_size:], axis = 1)
            Y_target = target.predict(X[batch_size:])
            Qsa = np.max(Y_target[actions], axis = 1).repeat(nb_actions)\
                                                     .reshape((batch_size, nb_actions))

        else:
            Qsa = np.max(Y[batch_size:], axis = 1).repeat(nb_actions)\
                                                .reshape((batch_size, nb_actions))

        # The targets here already take into account
        delta = np.zeros((batch_size, nb_actions))
        a = np.cast['int'](a)
        delta[np.arange(batch_size), a] = 1
        targets = ((1 - delta) * Y[:batch_size]
                  + delta * (r + gamma * (1 - game_over) * Qsa))

        if self.per: # Update the Sum Tree with the absolute error.
            errors = np.abs((targets - Y[:batch_size]).max(axis = 1))
            self.update(tree_indices, errors)

        return S, targets, IS_weights

    def reset_memory(self):
        """Set the memory as a blank list."""
        if self.per:
            if self.memory_size <= 0:
                self.memory_size = 150000

            self.memory = SumTree(self.memory_size)
            self.exp = 0
        else:
            self.memory = []


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

    def get_game_data(self, game, game_over):
        """Create a list with 4 frames and append/pop them each frame."""
        if game_over:
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

    def train(self, game, nb_epoch = 10000, batch_size = 64, gamma = 0.95,
              epsilon = [1., .01], epsilon_rate = 0.5, observe = 0,
              update_target_freq = 500, rounds = 1):
        """The main training function, loops the game, remember and choose best
        action given game state (frames)."""
        if type(epsilon)  in {tuple, list}:
            delta =  ((epsilon[0] - epsilon[1])\
                     / ((nb_epoch - observe) * epsilon_rate))
            final_epsilon = epsilon[1]
            epsilon = epsilon[0]
        else:
            final_epsilon = epsilon

        nb_actions = self.model.get_output_shape_at(0)[-1]
        win_count = 0
        for turn in range(rounds):
            if rounds > 1:
                self.reset_memory() # If more than one round, reset the memory

            for epoch in range(nb_epoch):
                loss = 0.
                game.reset()
                self.clear_frames()

                game_over = False
                S = self.get_game_data(game, game_over)

                while not game_over:
                    game.food_pos = game.generate_food()
                    rand = random.random()

                    if rand < epsilon or epoch < observe:
                        a = int(5 * rand) # Random action as often as epsilon.
                    else:
                        q = self.model.predict(S)
                        a = int(np.argmax(q[0]))

                    game.play(a, "ROBOT")
                    r = game.get_reward()

                    if game.snake.check_collision()\
                       or game.step > (50 * game.snake.length):
                       game_over = True # Cheeck collision before S'

                    S_prime = self.get_game_data(game, game_over)
                    experience = [S, a, r, S_prime, game_over]
                    self.memory.remember(*experience) # Add to the memory
                    S = S_prime # Advance to the next state (stack of S)

                    if epoch >= observe: # Get the batchs and train
                        batch = self.memory.get_targets(model = self.model,
                                                        target = self.target,
                                                        batch_size = batch_size,
                                                        gamma = gamma)

                        if batch:
                            inputs, targets, IS_weights = batch

                            if inputs is not None and targets is not None:
                                loss += float(self.model.train_on_batch(inputs,
                                                                        targets))

                if game.is_won():
                    win_count += 1 # Counter for metric purposes

                if epsilon > final_epsilon and epoch >= observe:
                    epsilon -= delta # Advance epsilon

                if self.per and self.memory.per_beta < 1.0: # Advance beta
                    self.memory.per_beta += self.memory.per_beta_inc

                if self.target is not None: # Update the target model
                    if epoch % update_target_freq == 0:
                        self.update_target_model()

                print("\tEpoch: {:03d}/{:03d} | Loss: {:.4f} | Epsilon: {:.2f}"\
                                                         .format(epoch + 1,
                                                                 nb_epoch, loss,
                                                                 epsilon)
                      + " | Win count: {} | Size: {:03d}".format(win_count,
                                                                 game.snake.length))

    def play(self, game, nb_epoch = 100, epsilon = 0., visual = False):
        """Play the game with the trained agent. Can use the visual tag to draw
            in pygame."""
        win_count = 0
        result_size = []
        result_step = []

        for epoch in range(nb_epoch):
            game.reset()
            self.clear_frames()
            game_over = False
            S = self.get_game_data(game, game_over)

            if visual:
                game.create_window()
                # The main loop, it pump key_presses and update every tick.
                environ['SDL_VIDEO_CENTERED'] = '1' # Centering the window
                previous_size = game.snake.length # Initial size of the snake
                color_list = game.gradient([(42, 42, 42), (152, 152, 152)],\
                                               previous_size)

            while not game_over:
                if epsilon != 0:
                    rand = random.random()

                    if rand < epsilon:
                        a = int(5 * rand) # Random action as often as epsilon.
                else:
                    q = self.model.predict(S)
                    a = int(np.argmax(q[0]))

                game.play(a, "ROBOT")
                current_size = game.snake.length # Update the body size

                if visual:
                    game.draw(color_list)

                    if current_size > previous_size:
                        color_list = game.gradient([(42, 42, 42), (152, 152, 152)],
                                                   game.snake.length)

                        previous_size = current_size

                if game.snake.check_collision()\
                    or game.step > (50 * current_size):
                    game_over = True

                S = self.get_game_data(game, game_over)

                if game_over:
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
                        local_state = arguments.local_state)
            agent = Agent(model = model, target = target, memory_size = -1,
                          nb_frames = nb_frames, board_size = board_size,
                          per = arguments.per)
            agent.train(game, batch_size = 64, nb_epoch = 10000, gamma = 0.8,
                        update_target_freq = update_target_freq)
        else:
            game = Game(board_size = board_size,
                        local_state = arguments.local_state)
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
                        local_state = arguments.local_state)
            agent = Agent(model = model, target = target, memory_size = -1,
                          nb_frames = nb_frames, board_size = board_size,
                          per = arguments.per)
            agent.train(game, batch_size = 64, nb_epoch = 10000, gamma = 0.8,
                        update_target_freq = update_target_freq)
        else:
            game = Game(board_size = board_size,
                        local_state = arguments.local_state)
            agent = Agent(model = model, target = target, memory_size = -1,
                          nb_frames = nb_frames, board_size = board_size,
                          per = arguments.per)

            print("Loading file located in {}. We can play after that."\
                  .format(arguments.args.load))

        print("--visual is activated. Drawing the board and controlled by the"
              + "DQN Agent. Playing:")
        agent.play(game, visual = True)
