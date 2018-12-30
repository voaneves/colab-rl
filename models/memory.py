"""TODO: Create a parent class to shorten code."""

import numpy as np
from random import sample, uniform
from array import array  # Efficient numeric arrays

from .utilities.sum_tree import *
from .utilities.misc import LinearSchedule


class Memory:
    """Base class for the experience replays.

    Methods
    ----------
        __init__
        reset_memory
        remember
        get_targets
    """
    def __init__(self,
                 memory_size = 1500000,
                 nb_epoch = 10000,
                 epsilon = 0.001,
                 alpha = 0.6,
                 beta = 0.4,
                 decay = 0.5,
                 prioritized = False):
        self.memory_size = memory_size
        self.prioritized = prioritized

        if self.prioritized:
            self.epsilon = epsilon
            self.alpha = alpha
            self.beta = beta
            self.schedule = LinearSchedule(nb_epoch * decay, 1.0, beta)
            self.max_priority = 1.0

        self.reset_memory()

    def reset_memory(self):
        """Set the memory as a blank list."""
        raise NotImplementedError

    def remember(self,
                 state,
                 action,
                 reward,
                 next_state,
                 game_over):
        raise NotImplementedError

    def get_targets(self,
                    target,
                    model,
                    batch_size,
                    nb_actions,
                    gamma = 0.95,
                    n_steps = 1):
        """Function to sample, set batch function and use it for targets."""
        if self._exp_size() < batch_size:
            return None

        samples, IS_weights, tree_indices = self._get_samples(batch_size)
        input_dim = np.prod(self.input_shape) # Get the input shape, multiplied

        S = samples[:, 0 : input_dim] # Seperate the states
        a = samples[:, input_dim] # Separate the actions
        r = samples[:, input_dim + 1] # Separate the rewards
        S_prime = samples[:, input_dim + 2 : 2 * input_dim + 2] # Next_actions
        game_over = samples[:, 2 * input_dim + 2] # Separate terminal flags

        # Reshape the arrays to make them usable by the model.
        S = S.reshape((batch_size, ) + self.input_shape)
        S_prime = S_prime.reshape((batch_size, ) + self.input_shape)

        X = np.concatenate([S, S_prime], axis = 0)
        Y = model.predict(X)

        if target is not None: # Use Double DQN logic:
            Qsa = [None] * 64
            actions = np.argmax(Y[batch_size:], axis = 1)
            Y_target = target.predict(X[batch_size:])

            for idx, target in enumerate(Y_target):
                Qsa[idx] = target[actions[idx]]

            Qsa = np.array(Qsa)
        else:
            Qsa = np.max(Y[batch_size:], axis = 1)

        # Where the action happened, replace with the Q values of S_prime
        targets = np.array(Y[:batch_size])
        value = r + (gamma ** n_steps) * (1 - game_over) * Qsa
        targets[range(batch_size), a.astype(int)] = value

        if self.prioritized:
            errors = np.abs(value - Y[:batch_size].max(axis = 1)).clip(max = 1.)
            self._update_priorities(tree_indices, errors)

        return S, targets, IS_weights

    def _format_experience(self,
                           state,
                           action,
                           reward,
                           next_state,
                           game_over):
        """Format SARS' experience and return it."""
        if not hasattr(self, 'input_shape'):
            self.input_shape = state.shape[1:] # set attribute only once

        experience = np.concatenate([state.flatten(),
                                     np.array(action).flatten(),
                                     np.array(reward).flatten(),
                                     next_state.flatten(),
                                     1 * np.array(game_over).flatten()])

        return experience

    def _exp_size(self):
        """Returns how much memory is stored."""
        raise NotImplementedError

    def __get_priority(self,
                       errors):
        """Returns priority based on how much prioritization to use."""
        return (errors + self.epsilon) ** self.alpha

    def _get_samples(self,
                     batch_size):
        raise NotImplementedError

    def _update_priorities(self,
                           tree_indices,
                           errors):
        raise NotImplementedError


class ExperienceReplay(Memory):
    """The class that handles memory and experiences replay.

    Attributes
    ----------
    memory: list of experiences
        Memory list to insert experiences.
    memory_size: int, optional, default = 150000
        The ammount of experiences to be stored in the memory.
    input_shape: tuple of 3 * int
        The shape of the input which will be stored.
    """
    def __init__(self,
                 memory_size = 150000):
        """Initialize parameters and the memory array."""
        super().__init__(memory_size = memory_size)

    def _exp_size(self):
        """Returns how much memory is stored."""
        return len(self.memory)

    def remember(self,
                 state,
                 action,
                 reward,
                 next_state,
                 game_over):
        """Remember SARS' experiences, with the game_over parameter (done)."""
        experience = super()._format_experience(state = state,
                                                action = action,
                                                reward = reward,
                                                next_state = next_state,
                                                game_over = game_over)
        self.memory.append(experience)

        if self.memory_size > 0 and self._exp_size() > self.memory_size:
            self.memory.pop(0)

    def _get_samples(self,
                     batch_size):
        """Sample the memory according to PER flag.

        Return
        ----------
        batch: np.array of batch_size experiences
            The batched experiences from memory.
        IS_weights: np.array of batch_size of the weights
            As it's used only in PER, is an array of ones in this case.
        Indexes: list of batch_size * int
            As it's used only in PER, return None.
        """
        IS_weights = np.ones((batch_size, ))
        batch = np.array(sample(self.memory, batch_size))

        return batch, IS_weights, None

    def get_targets(self,
                    target,
                    model,
                    batch_size,
                    nb_actions,
                    gamma = 0.9,
                    n_steps = 1):
        """Function to sample, set batch function and use it for targets."""
        batch = super().get_targets(target = target,
                                    model = model,
                                    batch_size = batch_size,
                                    nb_actions = nb_actions,
                                    gamma = gamma,
                                    n_steps = n_steps)

        return batch

    def reset_memory(self):
        """Set the memory as a blank list."""
        if self.memory_size <= 100:
            memory_size = 150000

        self.memory = []


class PrioritizedExperienceReplayNaive:
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
    def __init__(self,
                 memory_size = 150000,
                 nb_epoch = 10000,
                 epsilon = 0.001,
                 alpha = 0.6,
                 beta = 0.4,
                 decay = 0.5):
        """Initialize parameters and the memory array."""
        super().__init__(memory_size = memory_size,
                         nb_epoch = nb_epoch,
                         epsilon = epsilon,
                         alpha = alpha,
                         beta = beta,
                         decay = decay,
                         prioritized = True)

    def _exp_size(self):
        """Returns how much memory is stored."""
        return self.exp

    def remember(self,
                 state,
                 action,
                 reward,
                 next_state,
                 game_over):
        """Remember SARS' experiences, with the game_over parameter (done)."""
        experience = super()._format_experience(state = state,
                                                action = action,
                                                reward = reward,
                                                next_state = next_state,
                                                game_over = game_over)
        max_priority = self.memory.max_leaf()

        if max_priority == 0:
            max_priority = self._get_priority(0)

        self.memory.insert(experience, max_priority)
        self.exp += 1

    def _get_samples(self, batch_size):
        """Sample the memory according to PER flag."""
        batch = [None] * batch_size
        IS_weights = np.zeros((batch_size, ))
        tree_indices = [0] * batch_size

        memory_sum = self.memory.sum()
        len_seg = memory_sum / batch_size
        min_prob = self.memory.min_leaf() / memory_sum

        for i in range(batch_size):
            val = uniform(len_seg * i, len_seg * (i + 1))
            tree_indices[i], priority, batch[i] = self.memory.retrieve(val)
            prob = priority / self.memory.sum()
            IS_weights[i] = np.power(prob / min_prob, -self.beta)

        return np.array(batch), IS_weights, tree_indices

    def get_targets(self, target, model, batch_size, nb_actions, gamma = 0.9,
                    n_steps = 1):
        """Function to sample, set batch function and use it for targets."""
        batch = super().get_targets(target = target,
                                    model = model,
                                    batch_size = batch_size,
                                    nb_actions = nb_actions,
                                    gamma = gamma,
                                    n_steps = n_steps)

        return batch

    def _update_priorities(self, tree_indices, errors):
        """Update a list of nodes, based on their errors."""
        priorities = self._get_priority(errors)

        for index, priority in zip(tree_indices, priorities):
            self.memory.update(index, priority)

    def reset_memory(self):
        """Set the memory as a blank list."""
        if self.memory_size <= 100:
            self.memory_size = 150000

        self.memory = SumTree(self.memory_size)
        self.exp = 0


class PrioritizedExperienceReplay:
    def __init__(self,
                 memory_size = 1500000,
                 nb_epoch = 10000,
                 epsilon = 0.001,
                 alpha = 0.6,
                 beta = 0.4,
                 decay = 0.5):
        super().__init__(memory_size = memory_size,
                         nb_epoch = nb_epoch,
                         epsilon = epsilon,
                         alpha = alpha,
                         beta = beta,
                         decay = decay,
                         prioritized = True)

    def _exp_size(self):
        """Returns how much memory is stored."""
        return len(self.memory)

    def remember(self,
                 state,
                 action,
                 reward,
                 next_state,
                 game_over):
        """Remember SARS' experiences, with the game_over parameter (done)."""
        experience = super()._format_experience(state = state,
                                                action = action,
                                                reward = reward,
                                                next_state = next_state,
                                                game_over = game_over)

        if self._exp_size() < self.memory_size:
            self.memory.append(experience)
            self.pos += 1
        else:
            self.memory[self.pos] = experience
            self.pos = (self.pos + 1) % self.memory_size

        self._it_sum[self.pos] = self._get_priority(self.max_priority)
        self._it_min[self.pos] = self._get_priority(self.max_priority)

    def _sample_proportional(self, batch_size):
        proportional_indexes = array('i')

        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, self._exp_size() - 1)
            index = self._it_sum.find_prefixsum_idx(mass)
            proportional_indexes.append(index)

        return proportional_indexes

    def _get_samples(self, batch_size):
        proportional_indexes = self._sample_proportional(batch_size)

        weights = array('f')
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self._exp_size()) ** (-self.beta)

        for index in proportional_indexes:
            p_sample = self._it_sum[index] / self._it_sum.sum()
            weight = (p_sample * self._exp_size()) ** (-self.beta)
            weights.append(weight / max_weight)

        weights = np.array(weights, dtype=np.float32)
        samples = [self.memory[index] for index in proportional_indexes]

        return np.array(samples), weights, proportional_indexes

    def get_targets(self,
                    target,
                    model,
                    batch_size,
                    nb_actions,
                    gamma = 0.9,
                    n_steps = 1):
        """Function to sample, set batch function and use it for targets."""
        batch = super().get_targets(target = target,
                                    model = model,
                                    batch_size = batch_size,
                                    nb_actions = nb_actions,
                                    gamma = gamma,
                                    n_steps = n_steps)

        return batch

    def _update_priorities(self, tree_indices, errors):
        priorities = self._get_priority(errors)

        for index, priority in zip(tree_indices, priorities):
            self._it_sum[index] = priority ** self.alpha
            self._it_min[index] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def reset_memory(self):
        """Set the memory as a blank list."""
        if self.memory_size <= 100:
            self.memory_size = 150000

        self.memory = []
        self.pos = 0

        it_capacity = 1

        while it_capacity < self.memory_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
