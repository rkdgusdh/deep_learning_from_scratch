# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import time
import numpy as np
from common.optimizer import *


class RegressionTrainer:
    """Trainer for regression networks (the textbook common/trainer.py Trainer adapted for gaze estimation)

    - two inputs: (eye image, head pose)
    - per-epoch metric: mean angular error in degrees instead of accuracy
    - shuffles once per epoch and walks through mini-batches in order
    """
    def __init__(self, network, x_train, p_train, t_train, x_test, p_test, t_test,
                 epochs=10, mini_batch_size=100,
                 optimizer='Adam', optimizer_param={'lr': 0.001},
                 lr_decay=1.0, evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train, self.p_train, self.t_train = x_train, p_train, t_train
        self.x_test, self.p_test, self.t_test = x_test, p_test, t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.lr_decay = lr_decay
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer (same as the textbook Trainer)
        optimizer_class_dict = {'sgd': SGD, 'momentum': Momentum, 'nesterov': Nesterov,
                                'adagrad': AdaGrad, 'rmsprop': RMSprop, 'adam': Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size // mini_batch_size, 1)

        self.train_loss_list = []
        self.train_err_list = []
        self.test_err_list = []

    def _subset(self, x, p, t):
        n = self.evaluate_sample_num_per_epoch
        if n is None or n >= x.shape[0]:
            return x, p, t
        idx = np.linspace(0, x.shape[0] - 1, n).astype(int)
        return x[idx], p[idx], t[idx]

    def train_epoch(self, epoch):
        start = time.time()
        perm = np.random.permutation(self.train_size)
        for i in range(self.iter_per_epoch):
            batch_mask = perm[i * self.batch_size:(i + 1) * self.batch_size]
            x_batch = self.x_train[batch_mask]
            p_batch = self.p_train[batch_mask]
            t_batch = self.t_train[batch_mask]

            grads = self.network.gradient(x_batch, p_batch, t_batch)
            self.optimizer.update(self.network.params, grads)
            self.train_loss_list.append(self.network.last_layer.loss)  # reuse the loss from the forward pass inside gradient()

        train_err = self.network.angular_error(*self._subset(self.x_train, self.p_train, self.t_train))
        test_err = self.network.angular_error(*self._subset(self.x_test, self.p_test, self.t_test))
        self.train_err_list.append(train_err)
        self.test_err_list.append(test_err)
        if self.verbose:
            recent = np.mean(self.train_loss_list[-self.iter_per_epoch:])
            print("=== epoch:%d, loss:%.5f, train err:%.2f deg, test err:%.2f deg (%.0fs) ==="
                  % (epoch, recent, train_err, test_err, time.time() - start))

        if hasattr(self.optimizer, 'lr'):
            self.optimizer.lr *= self.lr_decay

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.train_epoch(epoch)

        test_err = self.network.angular_error(self.x_test, self.p_test, self.t_test)
        if self.verbose:
            print("=============== Final Test Angular Error ===============")
            print("test err: %.2f deg" % test_err)
        return test_err
