# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))  # allow importing the textbook common/ package
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from mpiigaze.layers_ext import MeanSquaredLoss, angular_error


class GazeNet:
    """Convolutional network for gaze estimation (LeNet-style, after Zhang et al., CVPR 2015)

    Extends the textbook's ch07 SimpleConvNet with:
      - non-square inputs (36x60 eye images)
      - two convolution blocks
      - head pose (2-D) concatenated before the last Affine layer
      - regression (MeanSquaredLoss) instead of classification (SoftmaxWithLoss)

    conv - relu - pool - conv - relu - pool - affine - relu - [concat pose] - affine - MSE

    Parameters
    ----------
    input_dim : input size (C, H, W)
    conv_params : list of conv layer settings [{'filter_num', 'filter_size', 'pad', 'stride'}, ...]
    hidden_size : number of neurons in the fully connected hidden layer
    pose_size : head pose input dimension (0 disables it)
    output_size : output size (pitch, yaw)
    """
    def __init__(self, input_dim=(1, 36, 60),
                 conv_params=({'filter_num': 20, 'filter_size': 5, 'pad': 0, 'stride': 1},
                              {'filter_num': 50, 'filter_size': 5, 'pad': 0, 'stride': 1}),
                 hidden_size=500, pose_size=2, output_size=2):
        C, H, W = input_dim
        self.pose_size = pose_size

        # initialize weights (He init, since we use ReLU)
        self.params = {}
        pre_channel = C
        for i, cp in enumerate(conv_params):
            idx = str(i + 1)
            fs = cp['filter_size']
            scale = np.sqrt(2.0 / (pre_channel * fs * fs))
            self.params['W' + idx] = scale * np.random.randn(cp['filter_num'], pre_channel, fs, fs)
            self.params['b' + idx] = np.zeros(cp['filter_num'])
            H = (H - fs + 2 * cp['pad']) // cp['stride'] + 1
            W = (W - fs + 2 * cp['pad']) // cp['stride'] + 1
            H, W = H // 2, W // 2  # 2x2 pooling
            pre_channel = cp['filter_num']
        flat_size = pre_channel * H * W

        n_conv = len(conv_params)
        self.fc1, self.fc2 = str(n_conv + 1), str(n_conv + 2)
        self.params['W' + self.fc1] = np.sqrt(2.0 / flat_size) * np.random.randn(flat_size, hidden_size)
        self.params['b' + self.fc1] = np.zeros(hidden_size)
        self.params['W' + self.fc2] = np.sqrt(1.0 / (hidden_size + pose_size)) * \
                                      np.random.randn(hidden_size + pose_size, output_size)
        self.params['b' + self.fc2] = np.zeros(output_size)

        # build layers
        self.layers = OrderedDict()
        for i, cp in enumerate(conv_params):
            idx = str(i + 1)
            self.layers['Conv' + idx] = Convolution(self.params['W' + idx], self.params['b' + idx],
                                                    cp['stride'], cp['pad'])
            self.layers['Relu' + idx] = Relu()
            self.layers['Pool' + idx] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W' + self.fc1], self.params['b' + self.fc1])
        self.layers['ReluFC'] = Relu()
        self.head = Affine(self.params['W' + self.fc2], self.params['b' + self.fc2])

        self.last_layer = MeanSquaredLoss()

    def predict(self, x, pose=None):
        for layer in self.layers.values():
            x = layer.forward(x)
        if self.pose_size > 0:
            x = np.concatenate([x, pose], axis=1)
        return self.head.forward(x)

    def loss(self, x, pose, t):
        y = self.predict(x, pose)
        return self.last_layer.forward(y, t)

    def angular_error(self, x, pose, t, batch_size=500):
        """Mean 3D angular error in degrees (the regression counterpart of accuracy)."""
        errors = []
        for i in range(0, x.shape[0], batch_size):
            y = self.predict(x[i:i + batch_size], None if pose is None else pose[i:i + batch_size])
            errors.append(angular_error(y, t[i:i + batch_size]))
        return float(np.mean(np.concatenate(errors)))

    def gradient(self, x, pose, t):
        # forward
        self.loss(x, pose, t)

        # backward
        dout = self.last_layer.backward(1)
        dout = self.head.backward(dout)
        dout = dout[:, :dout.shape[1] - self.pose_size]  # backward of concat: drop the gradient w.r.t. the pose input

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for i, key in enumerate(k for k in self.layers if k.startswith('Conv')):
            grads['W' + str(i + 1)] = self.layers[key].dW
            grads['b' + str(i + 1)] = self.layers[key].db
        grads['W' + self.fc1], grads['b' + self.fc1] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W' + self.fc2], grads['b' + self.fc2] = self.head.dW, self.head.db
        return grads

    def numerical_gradient(self, x, pose, t):
        from common.gradient import numerical_gradient
        loss_w = lambda w: self.loss(x, pose, t)
        return {key: numerical_gradient(loss_w, self.params[key]) for key in self.params}

    def save_params(self, file_name="gazenet_params.pkl"):
        with open(file_name, 'wb') as f:
            pickle.dump({k: v for k, v in self.params.items()}, f)

    def load_params(self, file_name="gazenet_params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key][...] = val  # copy in place because the layers reference the same arrays
