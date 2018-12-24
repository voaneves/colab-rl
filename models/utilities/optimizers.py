from __future__ import absolute_import

import numpy as np
from math import ceil, floor

from keras import backend as K
from keras.optimizers import Optimizer
from keras.legacy import interfaces

from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.framework import ops

if K.backend() == 'tensorflow':
    import tensorflow as tf


class COCOB(Optimizer):
    """COCOB-Backprop optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).
    This optimizer, unlike other stochastic gradient based optimizers, optimize the function by
    finding individual learning rates in a coin-betting way.
    # Arguments
        alphs: float >= 0. Multiples of the largest absolute magtitude of gradients.
        epsilon: float >= 0. Fuzz factor.
    # References
        - [Training Deep Networks without Learning Rates Through Coin Betting](http://https://arxiv.org/pdf/1705.07795.pdf)
    """

    def __init__(self, alpha=100, epsilon=1e-8, **kwargs):
        super(COCOB, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.alpha = K.variable(alpha, name='alpha')
            self.iterations = K.variable(0., name='iterations')
        self.epsilon = epsilon

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        L = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]
        M = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]
        Reward = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]
        grad_sum = [K.zeros(K.get_variable_shape(p), dtype=K.dtype(p)) for p in params]

        if K.eval(self.iterations) == 0:
            old_params = [K.constant(K.eval(p)) for p in params]
            # [K.eval(p) for p in params]

        self.weights = [self.iterations] + L + M + Reward + grad_sum

        for old_p, p, g, gs, l, m, r in zip(old_params, params, grads, grad_sum, L, M, Reward):
            # update accumulator
            # old_p = K.variable(old_p)

            new_l = K.maximum(l, K.abs(g))
            self.updates.append(K.update(l, new_l))

            new_m = m + K.abs(g)
            self.updates.append(K.update(m, new_m))

            new_r = K.maximum(r - (p - old_p)*g, 0)
            self.updates.append(K.update(r, new_r))

            new_gs = gs + g
            self.updates.append(K.update(gs, new_gs))

            new_p = old_p - (new_gs/(self.epsilon + new_l*K.maximum(new_m+new_l, self.alpha*new_l)))*(new_l + new_r)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'alpha': float(K.get_value(self.alpha)),
                  'epsilon': self.epsilon}
        base_config = super(COCOB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SMORMS3(Optimizer):
    '''SMORMS3 optimizer.
    Implemented based on http://sifter.org/~simon/journal/20150420.html
    # Arguments
        lr: float >= 0. Learning rate.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    '''

    def __init__(self, lr=0.001, epsilon=1e-16, decay=0.,
                 **kwargs):
        super(SMORMS3, self).__init__(**kwargs)
        self.__dict__.update(locals())
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr)
            # self.rho = K.variable(rho)
            self.decay = K.variable(decay)
            self.inital_decay = decay
            self.iterations = K.variable(0.)
        self.epsilon = epsilon
        self.initial_decay = decay

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.get_variable_shape(p) for p in params]
        self.updates.append(K.update_add(self.iterations, 1))

        g1s = [K.zeros(shape) for shape in shapes]
        g2s = [K.zeros(shape) for shape in shapes]
        mems = [K.ones(shape) for shape in shapes]

        lr = self.lr
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        self.weights = [self.iterations] + g1s + g2s + mems

        for p, g, g1, g2, m in zip(params, grads, g1s, g2s, mems):
            r = 1. / (m + 1)
            new_g1 = (1. - r) * g1 + r * g
            new_g2 = (1. - r) * g2 + r * K.square(g)
            # update accumulators
            self.updates.append(K.update(g1, new_g1))
            self.updates.append(K.update(g2, new_g2))
            new_p = p - g * K.minimum(lr, K.square(new_g1) / (new_g2 + self.epsilon)) / (
            K.sqrt(new_g2) + self.epsilon)
            new_m = 1 + m * (1 - K.square(new_g1) / (new_g2 + self.epsilon))
            # update rho
            self.updates.append(K.update(m, new_m))
            # apply constraints
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(SMORMS3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Yogi(Optimizer):
    """Yogi optimizer.
    Default parameters follow those provided in the original paper.
    Arguments:
      lr: float >= 0. Learning rate.
      beta_1: float, 0 < beta < 1. Generally close to 1.
      beta_2: float, 0 < beta < 1. Generally close to 1.
      epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
      decay: float >= 0. Learning rate decay over each update.
      amsgrad: boolean. Whether to apply the AMSGrad variant of this
          algorithm from the paper "On the Convergence of Adam and
          Beyond".
    """

    def __init__(self,
               lr=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=None,
               decay=0.00000001,
               amsgrad=False,
               **kwargs):
        super(Yogi, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (  # pylint: disable=g-no-augmented-assignment
                1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                    K.dtype(self.decay))))

        t = math_ops.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (
            K.sqrt(1. - math_ops.pow(self.beta_2, t)) /
            (1. - math_ops.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            #v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g) # from amsgrad
            v_t = v - (1-self.beta_2)*K.sign(v-math_ops.square(g))*math_ops.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
            }
        base_config = super(Yogi, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Nadamax(Optimizer):
    """Nesterov Adam optimizer with infinity norm.
    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.
    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, schedule_decay=0.004, **kwargs):
        super(Nadamax, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (K.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = K.maximum(self.beta_2 * v, K.abs(g))
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            p_t = p - self.lr * m_t_bar / (K.sqrt(v_t_prime) + self.epsilon)
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon,
                  'schedule_decay': self.schedule_decay}
        base_config = super(Nadamax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Radamax(Optimizer):
    """Nesterov Adam optimizer with infinity norm.
    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.
    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, schedule_decay=0.004, **kwargs):
        super(Radamax, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (K.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            if np.random.choice([1, -1]) == 1:
                v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            else:
                v_t = K.maximum(self.beta_2 * v, K.abs(g))
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            p_t = p - self.lr * m_t_bar / (K.sqrt(v_t_prime) + self.epsilon)
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'epsilon': self.epsilon,
                  'schedule_decay': self.schedule_decay}
        base_config = super(Radamax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdamDelta(Optimizer):
    """AdamDelta optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, rho=0.95,
                 epsilon=1e-8, decay=0., **kwargs):
        super(AdamDelta, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.rho = rho
            self.decay = K.variable(decay, name='decay')
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        delta_accumulators = [K.zeros(shape) for shape in shapes]

        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v, a, d_a in zip(params, grads, ms, vs, accumulators, delta_accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))

            # use the new accumulator and the *old* delta_accumulator
            update = g * K.sqrt(d_a + self.epsilon) / K.sqrt(new_a + self.epsilon)

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * update
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(update)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(AdamDelta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class YFOptimizer(object):
    """YellowFin optimizer, from https://github.com/nnormandin/YellowFin_Keras

    Must be used like this:
        opt = keras.optimizers.TFOptimizer(YFOptimizer())
    """
    def __init__(self, learning_rate=0.1, momentum=0.0, clip_thresh=None, beta=0.999, curv_win_width=20,
                 mu_update_interval=1, zero_debias=True, delta_mu=0.0):
        '''
    clip thresh is the threshold value on ||lr * gradient||
    delta_mu can be place holder/variable/python scalar. They are used for additional
    momentum in situations such as asynchronous-parallel training. The default is 0.0
    for basic usage of the optimizer.
    Args:
      lr: python scalar. The initial value of learning rate, we use 1.0 in our paper.
      mu: python scalar. The initial value of momentum, we use 0.0 in our paper.
      clip_thresh: python scalar. The cliping threshold for tf.clip_by_global_norm.
        if None, no clipping will be carried out.
      beta: python scalar. The smoothing parameter for estimations.
      delta_mu: for extensions. Not necessary in the basic use.
    Other features:
      If you want to manually control the learning rates, self.lr_factor is
      an interface to the outside, it is an multiplier for the internal learning rate
      in YellowFin. It is helpful when you want to do additional hand tuning
      or some decaying scheme to the tuned learning rate in YellowFin.
      Example on using lr_factor can be found here:
      https://github.com/JianGoForIt/YellowFin/blob/master/char-rnn-tensorflow/train_YF.py#L140
        '''
        self._lr = learning_rate
        self._mu = momentum

        self._lr_var = tf.Variable(learning_rate, dtype=tf.float32, name="YF_lr", trainable=False)
        self._mu_var = tf.Variable(momentum, dtype=tf.float32, name="YF_mu", trainable=False)
        # for step scheme or decaying scheme for the learning rates
        self.lr_factor = tf.Variable(1.0, dtype=tf.float32, name="YF_lr_factor", trainable=False)
        if clip_thresh is not None:
            self._clip_thresh_var = tf.Variable(clip_thresh, dtype=tf.float32, name="YF_clip_thresh", trainable=False)
        else:
            self._clip_thresh_var = None

        # the underlying momentum optimizer
        self._optimizer = \
          tf.train.MomentumOptimizer(self._lr_var * self.lr_factor, self._mu_var + delta_mu)

        # moving average for statistics
        self._beta = beta
        self._moving_averager = None

        # for global step counting
        self._global_step = tf.Variable(0, trainable=False)

        # for conditional tuning
        self._do_tune = tf.greater(self._global_step, tf.constant(0) )

        self._zero_debias = zero_debias

        self._tvars = None

        # for curvature range
        self._curv_win_width = curv_win_width
        self._curv_win = None


    def curvature_range(self):
        # set up the curvature window
        self._curv_win = \
          tf.Variable(np.zeros( [self._curv_win_width, ] ), dtype=tf.float32, name="curv_win", trainable=False)
        self._curv_win = tf.scatter_update(self._curv_win,
          self._global_step % self._curv_win_width, self._grad_norm_squared)
        # note here the iterations start from iteration 0
        valid_window = tf.slice(self._curv_win, tf.constant( [0, ] ),
          tf.expand_dims(tf.minimum(tf.constant(self._curv_win_width), self._global_step + 1), dim=0) )
        self._h_min_t = tf.reduce_min(valid_window)
        self._h_max_t = tf.reduce_max(valid_window)

        curv_range_ops = []
        with tf.control_dependencies([self._h_min_t, self._h_max_t] ):
            avg_op = self._moving_averager.apply([self._h_min_t, self._h_max_t] )
            with tf.control_dependencies([avg_op] ):
                self._h_min = tf.identity(self._moving_averager.average(self._h_min_t) )
                self._h_max = tf.identity(self._moving_averager.average(self._h_max_t) )
        curv_range_ops.append(avg_op)

        return curv_range_ops


    def grad_variance(self):
        grad_var_ops = []
        tensor_to_avg = []
        for t, g in zip(self._tvars, self._grads):
            if isinstance(g, ops.IndexedSlices):
                tensor_to_avg.append(tf.reshape(tf.unsorted_segment_sum(g.values, g.indices, g.dense_shape[0] ), shape=t.get_shape() ) )
            else:
                tensor_to_avg.append(g)
        avg_op = self._moving_averager.apply(tensor_to_avg)
        grad_var_ops.append(avg_op)
        with tf.control_dependencies([avg_op] ):
            self._grad_avg = [self._moving_averager.average(val) for val in tensor_to_avg]
            self._grad_avg_squared = [tf.square(val) for val in self._grad_avg]
        self._grad_var = self._grad_norm_squared_avg - tf.add_n( [tf.reduce_sum(val) for val in self._grad_avg_squared] )
        return grad_var_ops


    def dist_to_opt(self):
        dist_to_opt_ops = []
        # running average of the norm of gradeint
        self._grad_norm = tf.sqrt(self._grad_norm_squared)
        avg_op = self._moving_averager.apply([self._grad_norm,] )
        dist_to_opt_ops.append(avg_op)
        with tf.control_dependencies([avg_op] ):
            self._grad_norm_avg = self._moving_averager.average(self._grad_norm)
            # single iteration distance estimation, note here self._grad_norm_avg is per variable
            self._dist_to_opt = self._grad_norm_avg / self._grad_norm_squared_avg
        # running average of distance
        avg_op = self._moving_averager.apply([self._dist_to_opt] )
        dist_to_opt_ops.append(avg_op)
        with tf.control_dependencies([avg_op]):
            self._dist_to_opt_avg = tf.identity(self._moving_averager.average(self._dist_to_opt) )

        return dist_to_opt_ops


    def after_apply(self):
        self._moving_averager = tf.train.ExponentialMovingAverage(decay=self._beta, zero_debias=self._zero_debias)
        assert self._grads != None and len(self._grads) > 0
        after_apply_ops = []

        # get per var g**2 and norm**2
        self._grad_squared = []
        self._grad_norm_squared = []
        for v, g in zip(self._tvars, self._grads):
            with ops.colocate_with(v):
                self._grad_squared.append(tf.square(g) )
        self._grad_norm_squared = [tf.reduce_sum(grad_squared) for grad_squared in self._grad_squared]

        # the following running average on squared norm of gradient is shared by grad_var and dist_to_opt
        avg_op = self._moving_averager.apply(self._grad_norm_squared)
        with tf.control_dependencies([avg_op] ):
            self._grad_norm_squared_avg = [self._moving_averager.average(val) for val in self._grad_norm_squared]
            self._grad_norm_squared = tf.add_n(self._grad_norm_squared)
            self._grad_norm_squared_avg = tf.add_n(self._grad_norm_squared_avg)
        after_apply_ops.append(avg_op)

        with tf.control_dependencies([avg_op] ):
            curv_range_ops = self.curvature_range()
            after_apply_ops += curv_range_ops
            grad_var_ops = self.grad_variance()
            after_apply_ops += grad_var_ops
            dist_to_opt_ops = self.dist_to_opt()
            after_apply_ops += dist_to_opt_ops

        return tf.group(*after_apply_ops)


    def get_lr_tensor(self):
        lr = (1.0 - tf.sqrt(self._mu) )**2 / self._h_min
        return lr


    def get_mu_tensor(self):
        const_fact = self._dist_to_opt_avg**2 * self._h_min**2 / 2 / self._grad_var
        coef = tf.Variable([-1.0, 3.0, 0.0, 1.0], dtype=tf.float32, name="cubic_solver_coef")
        coef = tf.scatter_update(coef, tf.constant(2), -(3 + const_fact) )
        roots = tf.py_func(np.roots, [coef], Tout=tf.complex64, stateful=False)

        # filter out the correct root
        root_idx = tf.logical_and(tf.logical_and(tf.greater(tf.real(roots), tf.constant(0.0) ),
          tf.less(tf.real(roots), tf.constant(1.0) ) ), tf.less(tf.abs(tf.imag(roots) ), 1e-5) )
        # in case there are two duplicated roots satisfying the above condition
        root = tf.reshape(tf.gather(tf.gather(roots, tf.where(root_idx) ), tf.constant(0) ), shape=[] )
        tf.assert_equal(tf.size(root), tf.constant(1) )

        dr = self._h_max / self._h_min
        mu = tf.maximum(tf.real(root)**2, ( (tf.sqrt(dr) - 1)/(tf.sqrt(dr) + 1) )**2)
        return mu


    def update_hyper_param(self):
        assign_hyper_ops = []
        self._mu = tf.identity(tf.cond(self._do_tune, lambda: self.get_mu_tensor(),
          lambda: self._mu_var) )
        with tf.control_dependencies([self._mu] ):
            self._lr = tf.identity(tf.cond(self._do_tune, lambda: self.get_lr_tensor(),
            lambda: self._lr_var) )

        with tf.control_dependencies([self._mu, self._lr] ):
            self._mu = self._beta * self._mu_var + (1 - self._beta) * self._mu
            self._lr = self._beta * self._lr_var + (1 - self._beta) * self._lr
            assign_hyper_ops.append(tf.assign(self._mu_var, self._mu) )
            assign_hyper_ops.append(tf.assign(self._lr_var, self._lr) )
        assign_hyper_op = tf.group(*assign_hyper_ops)
        return assign_hyper_op


    def apply_gradients(self, grads_tvars, global_step):
        self._grads, self._tvars = zip(*grads_tvars)
        with tf.variable_scope("apply_updates"):
            if self._clip_thresh_var is not None:
                self._grads_clip, self._grads_norm = tf.clip_by_global_norm(self._grads, self._clip_thresh_var)
                apply_grad_op = \
                  self._optimizer.apply_gradients(zip(self._grads_clip, self._tvars) )
            else:
                apply_grad_op = \
                  self._optimizer.apply_gradients(zip(self._grads, self._tvars) )


        with tf.variable_scope("after_apply"):
            after_apply_op = self.after_apply()

        with tf.variable_scope("update_hyper"):
            with tf.control_dependencies( [after_apply_op] ):
                update_hyper_op = self.update_hyper_param()

        with tf.control_dependencies([update_hyper_op] ):
            self._increment_global_step_op = tf.assign(self._global_step, tf.cast(global_step, tf.int32))

        return tf.group(apply_grad_op, after_apply_op, update_hyper_op, self._increment_global_step_op)

    def compute_gradients(self, loss, var_list=None, gate_gradients=1,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        grad_loss=None):
        return(self._optimizer.compute_gradients(
        loss = loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss))


    def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=1, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
        """Adapted from Tensorflow Optimizer base class member function:
        Add operations to minimize `loss` by updating `var_list`.
        This method simply combines calls `compute_gradients()` and
        `apply_gradients()`. If you want to process the gradient before applying
        them call `tf.gradients()` and `self.apply_gradients()` explicitly instead
        of using this function.
        """
        grads_and_vars = self._optimizer.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
              "No gradients provided for any variable, check your graph for ops"
              " that do not support gradients, between variables %s and loss %s." %
              ([str(v) for _, v in grads_and_vars], loss))
        for g, v in grads_and_vars:
            print("g ", g)
            print("v ", v)

        return self.apply_gradients(grads_and_vars)
