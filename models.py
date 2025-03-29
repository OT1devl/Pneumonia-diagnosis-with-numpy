from functions import *
from utils import fast_convolution, fast_maxpool, fast_convolution_backprop, fast_maxpool_backprop

class SimpleCNN:
    def __init__(self, x, num_classes=2):
        self.input_shape = x.shape
        self.last_activation = sigmoid if num_classes <= 2 else softmax
        
        # Conv Layer 1

        self.conv_W0 = np.random.randn(3, 3, 1, 128) * np.sqrt(2/(3*3*1))
        self.conv_b0 = np.zeros((1, 1, 1, 128))
        
        self.conv_padding0 = 1
        self.conv_stride0 = 1
        self.conv_size0 = (self.input_shape[1], self.input_shape[2], 128)


        # Leaky ReLU 1

        # Max Pool 1

        self.pool1_size = 2
        self.pool1_stride = 1
        self.pool1_padding = 1

        # Conv Layer 2

        self.conv_W1 = np.random.randn(3, 3, 128, 64) * np.sqrt(2/(3*3*128))
        self.conv_b1 = np.zeros((1, 1, 1, 64))

        self.conv_padding1 = 1
        self.conv_stride1 = 1
        self.conv_size1 = (self.input_shape[1]//2, self.input_shape[2]//2, 64)

        # Leaky ReLU 2

        # Max Pool 2

        self.pool2_size = 2
        self.pool2_stride = 1
        self.pool2_padding = 1

        # Flatten 1
        dims = (self.input_shape[1]//4, self.input_shape[2]//4, 64)
        # Dense 1

        self.dense_W0 = np.random.randn(np.prod(dims), 64) * np.sqrt(2/np.prod(dims))
        self.dense_b0 = np.zeros((1, 64))

        # Leaky ReLU 3

        # Dense 2

        self.dense_W1 = np.random.randn(64, 16) * np.sqrt(2/64)
        self.dense_b1 = np.zeros((1, 16))

        # Leaky ReLU 4

        # Dense 3

        self.dense_W2 = np.random.randn(16, 1 if num_classes <= 2 else num_classes)
        self.dense_b2 = np.zeros((1, 1 if num_classes <= 2 else num_classes))

        # sigmoid or softmax

        # End

        self.params = [
            self.conv_W0, self.conv_b0,
            self.conv_W1, self.conv_b1,
            self.dense_W0, self.dense_b0,
            self.dense_W1, self.dense_b1,
            self.dense_W2, self.dense_b2
        ]
        self.len_params = len(self.params)

    def forward(self, x):
        self.conv_z0 = fast_convolution(x, self.conv_W0, self.conv_b0, padding=1, stride=1)
        self.conv_a0 = LeakyReLU(self.conv_z0)
        self.pool_conv_a0, self.pool_x_strided_0 = fast_maxpool(
            self.conv_a0, 2, 2, stride=1, padding=1
        )

        self.conv_z1 = fast_convolution(x, self.conv_W1, self.conv_b1, padding=1, stride=1)
        self.conv_a1 = LeakyReLU(self.conv_z1)
        self.pool_conv_a1, self.pool_x_strided_1 = fast_maxpool(
            self.conv_a1, 2, 2, stride=1, padding=1
        )

        self.flat = self.pool_conv_a1.reshape(self.pool_conv_a1.shape[0], -1)

        self.z0 = self.flat @ self.dense_W0 + self.dense_b0
        self.a0 = LeakyReLU(self.z0)

        self.z1 = self.a0 @ self.dense_W1 + self.dense_b1
        self.a1 = LeakyReLU(self.z1)

        self.z2 = self.a1 @ self.dense_W2 + self.dense_b2
        self.a2 = self.last_activation(self.z2)

        return self.a2
    
    def backward(self, x, y, outp, learn=True):
        dL = self.loss(y, outp, derv=True)

        if self.last_activation == sigmoid:
            dL *= sigmoid(self.a2, derv=True)

        m = dL.shape[0]

        dW2 = self.a1.T @ dL / m
        db2 = dL.sum(axis=0, keepdims=True) / m

        da1 = dL @ self.dense_W2.T
        dz1 = da1 * LeakyReLU(self.z1, derv=True)

        dW1 = self.a0.T @ dz1 / m
        db1 = dz1.sum(axis=0, keepdims=True) / m

        da0 = dz1 @ self.dense_W1.T
        dz0 = da0 * LeakyReLU(self.z0, derv=True)

        dW0 = self.flat.T @ dz0 / m
        db0 = dz0.sum(axis=0, keepdims=True) / m

        dflat = dz0 @ self.dense_W0.T

        dpool_conv_a1 = dflat.reshape(self.pool_conv_a1.shape)

        d_conv_a1 = fast_maxpool_backprop(self.conv_a1, self.pool2_size, self.pool2_size, self.pool2_stride, self.pool2_padding, dpool_conv_a1, self.pool_x_strided_1)

        d_conv_z1 = d_conv_a1 * LeakyReLU(self.conv_z1, derv=True)

        dx_conv1, dW2_conv, db2_conv = fast_convolution_backprop(self.pool_conv_a0, self.conv_W2, d_conv_z1, padding=self.conv_padding1, stride=self.conv_stride1)

        d_conv_a0 = fast_maxpool_backprop(self.conv_a0, self.pool1_size, self.pool1_size, self.pool1_stride, self.pool1_padding, dx_conv1, self.pool_x_strided_0)

        d_conv_z0 = d_conv_a0 * LeakyReLU(self.conv_z0, derv=True)

        dx_input, dW0_conv, db0_conv = fast_convolution_backprop(x, self.conv_W0, d_conv_z0, padding=self.conv_padding0, stride=self.conv_stride0)

        grads = dW0_conv, db0_conv, dW2_conv, db2_conv, dW0, db0, dW1, db1, dW2, db2

        if learn:
            for i in range(self.len_params):
                self.param -= self.optimizer.update_param(grads[i], i)

        # return dx_input, grads

    def compile(self, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

        self.optimizer.init_params(self.len_params)

    def train(self, x, y, epochs=50, batch_size=128, print_every=0.1, shuffle=True):
        losses = []
        accuracies = []
        for epoch in range(1, epochs+1):
            
            if shuffle:
                KEYS = np.arange(x.shape[0])
                np.random.shuffle(KEYS)
                x = x[KEYS]
                y = y[KEYS]
            
            epochs_loss = 0.0
            epochs_acc = 0.0
            
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                preds = self.forward(x_batch)
                loss = self.loss(y_batch, preds)
                acc = self.acc(y_batch, preds)
                epochs_loss += loss
                epochs_acc += acc

                self.optimizer.prev_step()
                self.backward(x_batch, y_batch, preds, learn=True)
                self.optimizer.step()
                print(f"Epoch {epoch}, Batch {i//batch_size + 1}", end='\r')
            
            avg_loss = epochs_loss / x.shape[0]
            avg_acc = epochs_acc / x.shape[0]

            losses.append(avg_loss)
            accuracies.append(avg_acc)

            if epoch % max(1, int(epochs * print_every)) == 0:
                print(f'Epoch: [{epoch}/{epochs}]> Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}')

        return losses, accuracies