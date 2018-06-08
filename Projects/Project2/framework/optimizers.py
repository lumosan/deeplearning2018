# -*- coding: utf-8 -*-

################ Generic class ################
class Optimizer(object):
    """
    Class for optimizers
    """
    def __init__(self):
        self.model = None

    def step(self, *input):
        raise NotImplementedError

    def adaptive_lr(kappa=0.75, eta0=1e-5):
        """Adaptive learning rate. After creating the lr with the
        values for kappa and eta0, it yields the value for the learning
        rate of the next iteration. Used for (Stochastic) Gradient Descent
        methods.
        """
        t = 1
        while True:
            yield eta0 * t ** -kappa
            t += 1



################ Implementations ################
class SGD(Optimizer):
    """Stochastic Gradient Descent with adaptive or fixed learning rate"""
    def __init__(self, a_lr=None):
        """
        INPUT
            a_lr: Expects None or a list of two elements.
                If not None, use the first parameter as kappa and the
                second as eta0 of the adaptive learning rate
        """
        if a_lr is not None:

            self.a_lr = Optimizer.adaptive_lr(kappa=a_lr[0], eta0=a_lr[1])
        else:
            self.a_lr = None

    def step(self, model, loss):
        """
        Performs one optimizer step, updating the gradient
        INPUT
            model
            loss: of last epoch
        """
        # If using adaptive learning rate, get next one
        if self.a_lr is not None:
            next_a_lr = next(self.a_lr)
        else:
            next_a_lr = None
        # Update gradients
        model.update(lr=next_a_lr)


class Adam(Optimizer):
    """Adam optimizer with fixed learning rate"""
    def __init__(self, alpha=1e-3, beta1=.9, beta2=.999, epsilon=1e-8):
        """
        INPUT
            alpha: learning rate
            beta1: exponential decay for first moment
            beta2: exponential decay for second moment
            epsilon: small number to avoid division by zero
        """
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1
        self.m_prev = None
        self.v_prev = None

    def restart(self):
        """Reset the parameters of the moments"""
        self.t = 1
        self.m_prev = None
        self.v_prev = None

    def compute_adam_moment_estimates(self, m_t_old, v_t_old, gradient):
        """
        Compute the new Adam moments
        INPUTS
            m_t_old: previous first moment
            v_t_old: previous second moment
            gradient
        """
        # compute bias-corrected first moment estimate
        m_t = (self.beta1 * m_t_old + (1 - self.beta1) * gradient)
        # compute bias-corrected second raw moment estimate
        v_t = (self.beta2 * v_t_old + (1 - self.beta2) * gradient.pow(2))

        out = (m_t / (1 - self.beta1 ** self.t) /
            ((v_t / (1 - self.beta2 ** self.t)).sqrt() + self.epsilon))

        return out, m_t, v_t

    def step(self, model, loss):
        """
        Performs one optimizer step, updating the gradient
        INPUT
            model
            loss: of last epoch
        """
        if self.m_prev is None:
            # 1st moment vector
            self.m_prev = [x[1].clone().fill_(0) for x in model.param()]
        if self.v_prev is None:
            # 2nd moment vector
            self.v_prev = [x[1].clone().fill_(0) for x in model.param()]

        # Compute moment estimates
        m_e = [self.compute_adam_moment_estimates(m_p, v_p, g[1]) for g, (m_p, v_p) in
            zip(model.param(), zip(self.m_prev, self.v_prev))]

        # Update optimizer with the computed values
        self.m_prev = [p[1] for p in m_e]
        self.v_prev = [p[2] for p in m_e]

        # Update parameters with the computed values
        out = [p[0] for p in m_e]
        model.update(lr=self.alpha, values=out)
        self.t += 1
