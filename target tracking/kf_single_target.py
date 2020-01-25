# Demonstration of Kalman Filter
# Credit to original MATLAB author:
# Maani Ghaffari, maanigj@umich.edu

# Python implementation author:
# Joseph Lowman, lowmanj@umich.edu
# Date: 1/25/2020

import numpy as np
from matplotlib import pyplot as plt

from kalman_filter import KalmanFilter

class GroundTruth:
    def __init__(self):
        self.x = np.linspace(-5.0, 5.0, num=101)
        self.y = np.sin(self.x) + 3.0

class SystemParameters:
    def __init__(self):
        self.A = np.eye(2)
        self.B = []
        self.H = np.eye(2)
        self.Q = 1E-3 * np.eye(2)
        self.R = 0.05**2 * np.eye(2)


class State:
    def __init__(self, x, sigma):
        self.x = x
        self.Sigma = sigma

np.random.seed(0)

ground_truth = GroundTruth()
system = SystemParameters()

# Measurement noise parameters
L = np.linalg.cholesky(system.R)

# Generate synthetic measurements with noise added to ground truth
z = np.vstack([ground_truth.x.copy(), ground_truth.y.copy()])
z += L @ np.random.random(z.shape)

x = np.array([[z[0, 0]],
              [z[1, 0]]])

initial_state = State(x, 2.0*np.eye(2))

kalman_filter = KalmanFilter(system, initial_state)

filtered_x = initial_state.x.copy() # keep track of Kalman state

for ind in range(1, z.shape[1]):
    kalman_filter.prediction()
    kalman_filter.correction(z[:, ind][:, None])

    filtered_x = np.hstack([filtered_x, kalman_filter.x])

fig, ax = plt.subplots(1, 1)
# plt.scatter(0, 0, , label="ownship")
plt.scatter([0], [0], marker="^", s=20, c="blue")
plt.plot(ground_truth.x, ground_truth.y, "-", label="ground truth")
plt.plot(filtered_x[0, :], filtered_x[1, :], "-k", label="Kalman filter")
plt.legend()

plt.xlabel("x_1")
plt.ylabel("x_2")
ax.set_aspect('equal')

plt.show()

