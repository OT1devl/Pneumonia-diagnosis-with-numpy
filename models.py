import numpy as np
from functions import *
from utils import fast_maxpool, fast_maxpool_backprop, fast_convolution, fast_convolution_backprop

DTYPE = np.float32

class Layer:
    count = 0
    def __init__(self, name=None):
        if name:
            self.name = name
        else:
            self.name = f'{self.__class__.__name__}_{__class__.count}'

        __class__.count+=1

    def give_grads(self):
        params = 'W', 'b'
        grads = 'dW', 'db'
        parameters = {}
        gradients = {}

        for key, val in self.__dict__.items():
            if key in params:
                parameters[f'{self.name}_{key}'] = val
            elif key in grads:
                gradients[f'{self.name}_{key}'] = val
        
        return parameters, gradients
    
class Conv2d(Layer):
    def __init__(self, channels_in, filters, kernel=(3, 3), padding=1, stride=1, name=None):
        super().__init__(name)
        self.W = np.random.randn(kernel[0], kernel[1], channels_in, filters) * np.sqrt(2/(kernel[0]*kernel[1]*channels_in))
        self.b = np.zeros((1, 1, 1, filters))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        self.inputs = x
        return fast_convolution(x, self.W, self.b, self.padding, self.stride)
    
    def backward(self, dout):
        dout, self.dW, self.db = fast_convolution_backprop(self.inputs, self.W, dout, self.padding, self.stride)
        return dout
    
class MaxPool2d(Layer):
    def __init__(self, pool_height, pool_width, padding=1, stride=1, name=None):
        super().__init__(name)
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        self.inputs = x
        out, self.x_strided = fast_maxpool(x, self.pool_height, self.pool_width, self.stride, self.padding)
        return out
    
    def backward(self, dout):
        return fast_maxpool_backprop(self.inputs, self.pool_height, self.pool_width, self.stride, self.padding, dout, self.x_strided)

class Dense(Layer):
    def __init__(self, neurons_in, neurons_out, name=None):
        super().__init__(name)
        self.W = np.random.randn(neurons_in, neurons_out) * np.sqrt(2/neurons_in)
        self.b = np.zeros((1, neurons_out))
    
    def forward(self, x):
        self.inputs = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, dout):
        self.dW = self.inputs.T @ dout / dout.shape[0]
        self.db = np.sum(dout, axis=0, keepdims=True) / dout.shape[0]

        return dout @ self.W.T

class SimpleCNN:
    def __init__(self, shapes, num_classes=2):
        self.last_activation = sigmoid if num_classes <= 2 else softmax

        self.conv1 = Conv2d(1, 4)
        # Leaky ReLU
        self.maxpool1 = MaxPool2d(2, 2)

        self.conv2 = Conv2d(4, 8)
        # Leaky ReLU
        self.maxpool2 = MaxPool2d(2, 2)

        self.conv3 = Conv2d(8, 16)
        # Leaky ReLU
        self.maxpool3 = MaxPool2d(2, 2)

        self.conv4 = Conv2d(16, 2)
        # Leaky ReLU
        self.maxpool4 = MaxPool2d(2, 2)

        # Flatten

        self.dense1 = Dense(26912, 64)
        # Leaky ReLU
        self.dense2 = Dense(64, 32)
        # Leaky ReLU
        self.dense3 = Dense(32, 1 if num_classes <= 2 else num_classes)
        # sigmoid or softmax

    def forward(self, x):
        # Primera etapa: Conv1 -> LeakyReLU -> MaxPool1
        self.conv_z0 = self.conv1.forward(x)
        self.conv_a0 = LeakyReLU(self.conv_z0)
        self.pool_conv_a0 = self.maxpool1.forward(self.conv_a0)

        # Segunda etapa: Conv2 -> LeakyReLU -> MaxPool2
        self.conv_z1 = self.conv2.forward(self.pool_conv_a0)
        self.conv_a1 = LeakyReLU(self.conv_z1)
        self.pool_conv_a1 = self.maxpool2.forward(self.conv_a1)

        # Tercera etapa: Conv3 -> LeakyReLU -> MaxPool3
        self.conv_z2 = self.conv3.forward(self.pool_conv_a1)
        self.conv_a2 = LeakyReLU(self.conv_z2)
        self.pool_conv_a2 = self.maxpool3.forward(self.conv_a2)

        # Cuarta etapa: Conv4 -> LeakyReLU -> MaxPool4
        self.conv_z3 = self.conv4.forward(self.pool_conv_a2)
        self.conv_a3 = LeakyReLU(self.conv_z3)
        self.pool_conv_a3 = self.maxpool4.forward(self.conv_a3)

        # Aplanado para conectar a las capas densas
        self.flat = self.pool_conv_a3.reshape(self.pool_conv_a3.shape[0], -1)

        # Capas densas: Dense1 -> LeakyReLU, Dense2 -> LeakyReLU y Dense3 -> (sigmoid o softmax)
        self.z0 = self.dense1.forward(self.flat)
        self.a0 = LeakyReLU(self.z0)

        self.z1 = self.dense2.forward(self.a0)
        self.a1 = LeakyReLU(self.z1)

        self.z2 = self.dense3.forward(self.a1)
        self.a2 = self.last_activation(self.z2)

        return self.a2

    def backward(self, y, outp):
        # Cálculo del gradiente de la pérdida
        dL = self.loss(y, outp, derv=True)
        if self.last_activation == sigmoid:
            # Suponiendo que la función sigmoid pueda calcular su derivada con la salida
            dL *= sigmoid(outp, derv=True)
        
        # Propagación hacia atrás por las capas densas
        dL = self.dense3.backward(dL)
        dL = LeakyReLU(self.z1, derv=True) * dL

        dL = self.dense2.backward(dL)
        dL = LeakyReLU(self.z0, derv=True) * dL

        dL = self.dense1.backward(dL)
        dL = dL.reshape(self.pool_conv_a3.shape)

        # Propagación hacia atrás por las capas convolucionales y de maxpooling (en orden inverso)
        dL = self.maxpool4.backward(dL)
        dL = LeakyReLU(self.conv_z3, derv=True) * dL
        dL = self.conv4.backward(dL)

        dL = self.maxpool3.backward(dL)
        dL = LeakyReLU(self.conv_z2, derv=True) * dL
        dL = self.conv3.backward(dL)

        dL = self.maxpool2.backward(dL)
        dL = LeakyReLU(self.conv_z1, derv=True) * dL
        dL = self.conv2.backward(dL)

        dL = self.maxpool1.backward(dL)
        dL = LeakyReLU(self.conv_z0, derv=True) * dL
        dL = self.conv1.backward(dL)

        return dL

    
    def compile(self, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    def update_params(self):
        layers = self.conv1, self.conv2, self.conv3, self.conv4, self.dense1, self.dense2, self.dense3
        self.optimizer.prev_update()
        for layer in layers:
            self.optimizer.update(*layer.give_grads())
        self.optimizer.step()

    def train(self, x, y, epochs=10, batch_size=16, print_every=0.1):
        losses = []
        accuracies = []
        for epoch in range(1, epochs + 1):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x = x[indices]
            y = y[indices]
            
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_batches = 0
            
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                preds = self.forward(x_batch)
                loss = self.loss(y_batch, preds)
                acc = self.accuracy(y_batch, preds)
                epoch_loss += loss
                epoch_acc += acc
                num_batches += 1
                
                self.backward(y_batch, outp=preds)
                self.update_params()
                print(f"Epoch {epoch}, Batch {i//batch_size + 1}", end='\r')
            
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches
            losses.append(avg_loss)
            accuracies.append(avg_acc)
            if epoch % max(1, int(epochs * print_every)) == 0:
                print(f'Epoch: [{epoch}/{epochs}]> Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')
        return losses, accuracies