import random
import numpy as np

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class GreedyQPolicy:
    """Implement the greedy policy

    Greedy policy always takes current best action.
    """
    def __init__(self):
        super(GreedyQPolicy, self).__init__()

    def select_action(self, model, state, epoch, nb_actions):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        q = model.predict(state)
        action = int(np.argmax(q[0]))

        return action, 0

    def get_config(self):
        """Return configurations of EpsGreedyPolicy
        # Returns
            Dict of config
        """
        config = super(GreedyQPolicy, self).get_config()
        return config


class EpsGreedyQPolicy:
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """
    def __init__(self, max_eps=1., min_eps = .01, nb_epoch = 10000):
        super(EpsGreedyQPolicy, self).__init__()
        self.schedule = LinearSchedule(nb_epoch, min_eps, max_eps)

    def select_action(self, model, state, epoch, nb_actions):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        rand = random.random()
        self.eps = self.schedule.value(epoch)

        if rand < self.eps:
            action = int(nb_actions * rand)
        else:
            q = model.predict(state)
            action = int(np.argmax(q[0]))

        return action, self.eps

    def get_config(self):
        """Return configurations of EpsGreedyPolicy
        # Returns
            Dict of config
        """
        config = super(EpsGreedyQPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class BoltzmannQPolicy:
    """Implement the Boltzmann Q Policy
    Boltzmann Q Policy builds a probability law on q values and returns
    an action selected randomly according to this law.
    """
    def __init__(self, max_temp = 1., min_temp = .01, nb_epoch = 10000, clip = (-500., 500.)):
        super(BoltzmannQPolicy, self).__init__()
        self.schedule = LinearSchedule(nb_epoch, min_temp, max_temp)
        self.clip = clip

    def select_action(self, model, state, epoch, nb_actions):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        q = model.predict(state)[0]
        self.temp = self.schedule.value(epoch)
        arg = q / self.temp

        exp_values = np.exp(arg - arg.max())
        probs = exp_values / exp_values.sum()
        action = np.random.choice(range(nb_actions), p = probs)

        return action, self.temp

    def get_config(self):
        """Return configurations of EpsGreedyPolicy
        # Returns
            Dict of config
        """
        config = super(BoltzmannQPolicy, self).get_config()
        config['temp'] = self.temp
        config['clip'] = self.clip
        return config


class BoltzmannGumbelQPolicy:
    """Implements Boltzmann-Gumbel exploration (BGE) adapted for Q learning
    based on the paper Boltzmann Exploration Done Right
    (https://arxiv.org/pdf/1705.10257.pdf).
    BGE is invariant with respect to the mean of the rewards but not their
    variance. The parameter C, which defaults to 1, can be used to correct for
    this, and should be set to the least upper bound on the standard deviation
    of the rewards.
    BGE is only available for training, not testing. For testing purposes, you
    can achieve approximately the same result as BGE after training for N steps
    on K actions with parameter C by using the BoltzmannQPolicy and setting
    tau = C/sqrt(N/K)."""

    def __init__(self, C = 1.0):
        super(BoltzmannGumbelQPolicy, self).__init__()
        self.C = C
        self.action_counts = None

    def select_action(self, model, state, epoch, nb_actions):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        q = model.predict(state)[0]
        q = q.astype('float64')

        # If we are starting training, we should reset the action_counts.
        # Otherwise, action_counts should already be initialized, since we
        # always do so when we begin training.
        if epoch == 0:
            self.action_counts = np.ones(q.shape)

        beta = self.C/np.sqrt(self.action_counts)
        Z = np.random.gumbel(size = q.shape)

        perturbation = beta * Z
        perturbed_q_values = q + perturbation
        action = np.argmax(perturbed_q_values)

        self.action_counts[action] += 1
        return action, np.sum(self.action_counts)

    def get_config(self):
        """Return configurations of EpsGreedyPolicy
        # Returns
            Dict of config
        """
        config = super(BoltzmannGumbelQPolicy, self).get_config()
	    config['C'] = self.C
        return config
