#!/usr/bin/env python

"""simple_dqn: First try to create a AI for SnakeGame. Is it good enough?

Arguments:
    --load FILE.h5: load a previously trained model in '.h5' format.
    --board_size INT: assign the size of the board, a INT type.
    --visual OPTIONAL: select wheter or not to draw the game in pygame.
"""


import numpy as np
from os import path, environ, sys
from random import sample, random

import inspect # Making relative imports from parallel folders possible
currentdir = path.dirname(path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = path.dirname(currentdir)
sys.path.insert(0, parentdir)

from game.snake import Game
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from utilities.utilities import *

__author__ = "Victor Neves"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Victor Neves"
__email__ = "victorneves478@gmail.com"
__status__ = "Production"

K.set_image_dim_ordering('th')
board_size = 10
nb_frames = 4
nb_actions = 5

class ExperienceReplay:
    """The class that handles memory and experiences replay.

    Attributes:
        fast: faster batch importing.
        memory: memory array to insert experiences.
        memory_size: the ammount of experiences to be stored in the memory.
        input_shape: the shape of the input which will be stored.
        batch_function: returns targets according to S.
    """
    def __init__(self, memory_size = 100, fast = True):
        """Initialize parameters and the memory array."""
        self.fast = fast
        self._memory_size = memory_size
        self.reset_memory()

    def remember(self, s, a, r, s_prime, game_over):
        """Remember SARS' experiences, with the game_over parameter (done)."""
        self.input_shape = s.shape[1:]
        self.memory.append(np.concatenate([s.flatten(),
                                           np.array(a).flatten(),
                                           np.array(r).flatten(),
                                           s_prime.flatten(),
                                           1 * np.array(game_over).flatten()]))

        if self.memory_size > 0 and len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def get_batch(self, model, batch_size, gamma = 0.9):
        """Function to sample, set batch function and use it for targets."""
        if len(self.memory) < batch_size:
            return None

        samples = np.array(sample(self.memory, batch_size))
        if not hasattr(self, 'batch_function'):
            self.set_batch_function(model, self.input_shape, batch_size,
                                    model.get_output_shape_at(0)[-1], gamma)

        S, targets = self.batch_function([samples])

        return S, targets

    @property
    def memory_size(self):
        return self._memory_size

    @memory_size.setter
    def memory_size(self, value):
        if value > 0 and value < self._memory_size:
            self.memory = self.memory[:value]

        self._memory_size = value

    def reset_memory(self):
        """Set the memory as a blank list."""
        self.memory = []

    def set_batch_function(self, model, input_shape, batch_size,
                           nb_actions, gamma):
        """Set the batch function that returns targets."""
        input_dim = np.prod(input_shape)
        samples = K.placeholder(shape = (batch_size, input_dim * 2 + 3))
        S = samples[:, 0 : input_dim]
        a = samples[:, input_dim]
        r = samples[:, input_dim + 1]
        S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2]
        game_over = samples[:, 2 * input_dim + 2 : 2 * input_dim + 3]
        r = K.reshape(r, (batch_size, 1))
        r = K.repeat(r, nb_actions)
        r = K.reshape(r, (batch_size, nb_actions))
        game_over = K.repeat(game_over, nb_actions)
        game_over = K.reshape(game_over, (batch_size, nb_actions))
        S = K.reshape(S, (batch_size, ) + input_shape)
        S_prime = K.reshape(S_prime, (batch_size, ) + input_shape)
        X = K.concatenate([S, S_prime], axis = 0)
        Y = model(X)
        Qsa = K.max(Y[batch_size:], axis = 1)
        Qsa = K.reshape(Qsa, (batch_size, 1))
        Qsa = K.repeat(Qsa, nb_actions)
        Qsa = K.reshape(Qsa, (batch_size, nb_actions))
        delta = K.reshape(self.one_hot(a, nb_actions), (batch_size, nb_actions))
        targets = (1 - delta) * Y[:batch_size] + delta * (r + gamma * (1 - game_over) * Qsa)
        self.batch_function = K.function(inputs = [samples], outputs = [S, targets])

    def one_hot(self, seq, num_classes):
        """Hot encoding for a seq, according to number of classes."""
        return K.one_hot(K.reshape(K.cast(seq, "int32"), (-1, 1)), num_classes)


class Agent:
	"""Agent based in a simple DQN that can read states, remember and play.

	Attributes:
		memory: memory used in the model. Input memory or ExperienceReplay.
		model: the input model, Conv2D in Keras.
		nb_frames: ammount of frames for each sars.
		frames: the frames in each sars.
	"""
	def __init__(self, model, memory = None, memory_size = 1000,
				 nb_frames = None):
		"""Initialize the agent with given attributes."""
		if memory:
			self.memory = memory
		else:
			self.memory = ExperienceReplay(memory_size)

		self.model = model
		self.nb_frames = nb_frames
		self.frames = None

	@property
	def memory_size(self):
		return self.memory.memory_size

	@memory_size.setter
	def memory_size(self, value):
		self.memory.memory_size = value

	def reset_memory(self):
		"""Reset memory if necessary."""
		self.exp_replay.reset_memory()

	def get_game_data(self, game):
		"""Create a list with 4 frames and append/pop them each frame."""
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

	def train(self, game, nb_epoch = 1000, batch_size = 50, gamma = 0.9,
			  epsilon = [1., .01], epsilon_rate = 0.5, reset_memory = False,
			  observe = 0):
		"""The main training function, loops the game, remember and choose best
		action given game state (frames)."""
		if type(epsilon)  in {tuple, list}:
			delta =  ((epsilon[0] - epsilon[1]) / (nb_epoch * epsilon_rate))
			final_epsilon = epsilon[1]
			epsilon = epsilon[0]
		else:
			final_epsilon = epsilon

		nb_actions = self.model.get_output_shape_at(0)[-1]
		win_count = 0

		for epoch in range(nb_epoch):
			loss = 0.
			game.reset()
			self.clear_frames()

			if reset_memory:
				self.reset_memory()

			game_over = False
			S = self.get_game_data(game)

			while not game_over:
				game.food_pos = game.generate_food()
				rand = np.random.random()

				if rand < epsilon or epoch < observe:
					a = int(5 * rand)
				else:
					q = self.model.predict(S)
					a = int(np.argmax(q[0]))

				game.play(a, "ROBOT")
				r = game.get_reward()
				S_prime = self.get_game_data(game)
				if game.snake.check_collision()\
				   or game.step > (100 * game.snake.length):
					game_over = True
				transition = [S, a, r, S_prime, game_over]
				self.memory.remember(*transition)
				S = S_prime

				if epoch >= observe:
					batch = self.memory.get_batch(model = model,
												  batch_size = batch_size,
												  gamma = gamma)

					if batch:
						inputs, targets = batch
						if inputs is not None and targets is not None:
							loss += float(model.train_on_batch(inputs, targets))

			if game.is_won():
				win_count += 1

			if epsilon > final_epsilon and epoch >= observe:
				epsilon -= delta

			print("\tEpoch: {:03d}/{:03d} | Loss: {:.4f} | Epsilon: {:.2f} | Win count: {} | Size: {:03d}"\
              	  .format(epoch + 1, nb_epoch, loss, epsilon, win_count, game.snake.length))

	def play(self, game, nb_epoch = 100, epsilon = 0., visual = False):
		"""Play the game with the trained agent. Can use the visual tag to draw
        in pygame."""
		model = self.model
		win_count = 0
		result_size = []
		result_step = []

		for epoch in range(nb_epoch):
			game.reset()
			self.clear_frames()
			S = self.get_game_data(game)
			game_over = False
			current_size = 3 # Initial size of the snake

			if visual:
				game.create_window()
				# The main loop, it pump key_presses and update every tick.
				environ['SDL_VIDEO_CENTERED'] = '1' # Centering the window
				previous_size = game.snake.length # Initial size of the snake
				color_list = game.gradient([(42, 42, 42), (152, 152, 152)],\
			                               previous_size)
				filename = []

			while not game_over:
				q = model.predict(S)[0]
				possible_actions = range(0, 5)
				q = [q[i] for i in possible_actions]
				action = possible_actions[np.argmax(q)]

				game.play(action, "ROBOT")
				S = self.get_game_data(game)
				current_size = game.snake.length # Update the body size

				if visual:
					game.draw(color_list)

					if current_size > previous_size:
						color_list = game.gradient([(42, 42, 42),
													(152, 152, 152)],
			                                       	game.snake.length)

					previous_size = current_size
				if game.snake.check_collision()\
				   or game.step > (100 * game.snake.length):
					game_over = True

				if game_over:
					result_size.append(current_size)
					result_step.append(game.step)

			if game.is_won():
				win_count += 1

		print("Accuracy: {} %".format(100. * win_count / nb_epoch))
		print("Mean size: {}".format(np.mean(result_size)))
		print("Mean steps: {}".format(np.mean(result_step)))

if __name__ == '__main__':
    arguments = HandleArguments()

    if not arguments.status_visual:
        if not arguments.status_load:
            model = CNN1(optimizer = RMSprop(), loss = clipped_error,
                        stack = nb_frames, input_size = board_size,
                        output_size = nb_actions)

            print("Not using --load. Default behavior is to train the model and then play. Training:")

            game = Game(board_size = board_size)
            agent = Agent(model = model, memory_size = 150000, nb_frames = nb_frames)
            agent.train(game, batch_size = 64, nb_epoch = 10000, gamma = 0.8)
        else:
            game = Game(board_size = board_size)
            agent = Agent(model = model, memory_size = 150000, nb_frames = nb_frames)

            print("Loading file located in {}. We can play after that." .format(arguments.args.load))

        print("--visual not used. Default behavior is to have drawing disabled. Playing:")
        agent.play(game, visual = False)
    else:
        if not arguments.status_load:
            model = CNN1(optimizer = RMSprop(), loss = clipped_error,
                        stack = nb_frames, input_size = board_size,
                        output_size = nb_actions)

            print("Not using --load. Default behavior is to train the model and then play. Training:")

            game = Game(board_size = board_size)
            agent = Agent(model = model, memory_size = 150000, nb_frames = nb_frames)
            agent.train(game, batch_size = 64, nb_epoch = 10000, gamma = 0.8)
        else:
            game = Game(board_size = board_size)
            agent = Agent(model = model, memory_size = 150000, nb_frames = nb_frames)

            print("Loading file located in {}. We can play after that." .format(arguments.args.load))

        print("--visual is activated. Drawing the board and controlled by the DQN Agent. Playing:")
        agent.play(game, visual = True)