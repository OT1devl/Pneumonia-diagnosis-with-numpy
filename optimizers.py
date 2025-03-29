import numpy as np

class Adam:
    def __init__(self, lr=0.001, decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.lr = lr
        self.current_lr = lr
        self.decay = decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 1

    def init_params(self, total_params):
        self.m = [0] * total_params
        self.v = [0] * total_params

    def prev_step(self):
        if self.decay:
            self.current_lr = self.lr * (1 /(1 + self.decay * self.t))

    def update_params(self, grads, i):
        self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grads
        self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * grads**2
        m_h = self.m[i] / (1 - self.beta_1**self.t)
        v_h = self.v[i] / (1 - self.beta_2**self.t)
        return self.current_lr * (m_h / (np.sqrt(v_h) + self.epsilon))

    def step(self):
        self.t += 1