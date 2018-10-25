import numpy as np

from random import sample, uniform
from utilities.sum_tree import SumTree
from utilities.policy import LinearSchedule

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
            self.schedule = LinearSchedule(nb_epoch * decay, 1.0, beta)

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
            tree_indices = [0] * batch_size

            memory_sum = self.memory.sum()
            len_seg = memory_sum / batch_size
            min_prob = self.memory.min_leaf() / memory_sum

            for i in range(batch_size):
                val = uniform(len_seg * i, len_seg * (i + 1))
                tree_indices[i], priority, batch[i] = self.memory.retrieve(val)
                prob = priority / self.memory.sum()
                IS_weights[i] = np.power(prob / min_prob, -self.per_beta)

            return np.array(batch), IS_weights, tree_indices

        else:
            IS_weights = np.ones((batch_size, ))
            batch = sample(self.memory, batch_size)
            return np.array(batch), IS_weights, None

    def get_targets(self, target, model, batch_size, nb_actions, gamma = 0.9):
        """Function to sample, set batch function and use it for targets."""
        if self.exp_size() < batch_size:
            return None

        samples, IS_weights, tree_indices = self.get_samples(batch_size)
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
            Qsa = [None] * 64
            actions = np.argmax(Y[batch_size:], axis = 1)
            Y_target = target.predict(X[batch_size:])
            for i in range(batch_size):
                Qsa[i] = Y_target[i][actions[i]]
            Qsa = np.array(Qsa).repeat(nb_actions).reshape((batch_size, nb_actions))

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
            errors = np.abs((targets - Y[:batch_size]).max(axis = 1)).clip(max = 1.)
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
