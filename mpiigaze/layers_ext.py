# coding: utf-8
"""Regression layer missing from the textbook common/layers.py, plus gaze-angle utilities."""
import numpy as np


class MeanSquaredLoss:
    """Mean squared error layer (same interface as the textbook SoftmaxWithLoss).

    loss = 0.5 * sum((y - t)^2) / N
    """
    def __init__(self):
        self.loss = None
        self.y = None   # network output
        self.t = None   # target (pitch, yaw)

    def forward(self, x, t):
        self.y = x
        self.t = t
        batch_size = x.shape[0]
        self.loss = 0.5 * np.sum((x - t) ** 2) / batch_size
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        return dout * (self.y - self.t) / batch_size


def pitchyaw_to_vector(pitchyaw):
    """(N, 2) [pitch, yaw] in radians -> (N, 3) unit gaze vectors.

    MPIIGaze normalized space: looking straight at the camera is the -z direction.
    Inverse of pitch = arcsin(-y), yaw = arctan2(-x, -z).
    """
    pitch, yaw = pitchyaw[:, 0], pitchyaw[:, 1]
    x = -np.cos(pitch) * np.sin(yaw)
    y = -np.sin(pitch)
    z = -np.cos(pitch) * np.cos(yaw)
    return np.stack([x, y, z], axis=1)


def vector_to_pitchyaw(vectors):
    """(N, 3) gaze vectors -> (N, 2) [pitch, yaw] in radians."""
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    pitch = np.arcsin(-v[:, 1])
    yaw = np.arctan2(-v[:, 0], -v[:, 2])
    return np.stack([pitch, yaw], axis=1)


def angular_error(pred, true):
    """Per-sample 3D angular error in degrees between predicted and true (pitch, yaw)."""
    a = pitchyaw_to_vector(pred)
    b = pitchyaw_to_vector(true)
    cos = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))
