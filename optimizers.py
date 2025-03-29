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
        self.m = {}
        self.v = {}

    def prev_update(self):
        if self.decay:
            self.current_lr = self.lr * (1 / (1 + self.decay * self.t))

    def update(self, params: dict, grads: dict):
        if self.t == 1:
            for key, val in params.items():
                if key not in self.m:
                    self.m[key] = np.zeros_like(val)
                if key not in self.v:
                    self.v[key] = np.zeros_like(val)

        for param_name, grad_name in zip(params.keys(), grads.keys()):
            self.m[param_name] = self.beta_1 * self.m[param_name] + (1 - self.beta_1) * grads[grad_name]
            self.v[param_name] = self.beta_2 * self.v[param_name] + (1 - self.beta_2) * grads[grad_name]**2
            m_h = self.m[param_name] / (1 - self.beta_1**self.t)
            v_h = self.v[param_name] / (1 - self.beta_2**self.t)
            params[param_name] -= self.current_lr * (m_h / (np.sqrt(v_h) + self.epsilon))

    def step(self):
        self.t+=1