import numpy as np

# Activations

def ReLU(x, derv=False):
    if derv: return np.where(x>0, 1, 0)
    return np.maximum(x, 0)

def LeakyReLU(x, alpha=0.2, derv=False):
    if derv: return np.where(x>0, 1, alpha)
    return np.where(x>0, x, x*alpha)

def sigmoid(x, derv=False):
    if derv: return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

# Losses

def BCE(y_true, y_pred, epsilon=1e-8, derv=False):
    if derv: return -y_true / (y_pred + epsilon) + (1 - y_true) / (1 - y_pred + epsilon)
    return np.mean(-y_true * np.log(y_pred + epsilon) - (1 - y_true) * np.log(1 - y_pred + epsilon))

def CCE(y_true, y_pred, epsilon=1e-8, derv=False):
    if derv: return y_pred - y_true
    return -np.mean(y_true * np.log(y_pred + epsilon))

# Accuracies

def SingleBinaryAccuracy(y_true, y_preds):
    return np.mean((y_preds > 0.5) == y_true)

def SingleCategoricalAccuracy(y_true, y_preds):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_preds, axis=1))

def BinaryAccuracy(y, x, model):
    batch_size = 128
    total = x.shape[0]
    loss_cum = 0.0
    acc_cum = 0.0
    
    for batch in range(0, x.shape[0], batch_size):
        x_batch = x[batch:batch+batch_size]
        y_batch = y[batch:batch+batch_size]
        predictions = model.forward(x_batch)
        loss_cum += BCE(y_batch, predictions)
        acc_cum += np.sum((predictions > 0.5)==y_batch)

    return acc_cum/total, loss_cum/total

def CategoricalAccuracy(y, x, model):
    batch_size = 128
    total = x.shape[0]
    loss_cum = 0.0
    acc_cum = 0.0
    
    for batch in range(0, x.shape[0], batch_size):
        x_batch = x[batch:batch+batch_size]
        y_batch = y[batch:batch+batch_size]
        predictions = model.forward(x_batch)
        loss_cum += CCE(y_batch, predictions)
        acc_cum += np.sum(np.argmax(y_batch, axis=1) == np.argmax(predictions, axis=1))

    return acc_cum/total, loss_cum/total