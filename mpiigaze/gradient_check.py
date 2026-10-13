# coding: utf-8
"""Compare numerical gradients with backprop gradients (same approach as textbook ch05/gradient_check.py)"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import numpy as np
from mpiigaze.gaze_net import GazeNet

np.random.seed(0)
# numerical differentiation is slow, so check with a tiny network and tiny inputs
network = GazeNet(input_dim=(1, 12, 16),
                  conv_params=({'filter_num': 3, 'filter_size': 3, 'pad': 0, 'stride': 1},
                               {'filter_num': 4, 'filter_size': 3, 'pad': 0, 'stride': 1}),
                  hidden_size=10)
x = np.random.rand(3, 1, 12, 16)
pose = np.random.randn(3, 2) * 0.2
t = np.random.randn(3, 2) * 0.2

grad_numerical = network.numerical_gradient(x, pose, t)
grad_backprop = network.gradient(x, pose, t)

max_diff = 0.0
for key in sorted(grad_numerical):
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    max_diff = max(max_diff, diff)
    print(key + ":" + str(diff))

assert max_diff < 1e-6, "gradient mismatch: %g" % max_diff
print("OK (max diff %.2e)" % max_diff)
