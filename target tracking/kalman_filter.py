import numpy as np

class KalmanFilter:
    def __init__(self, system_params, initial_state):
        self.A = system_params.A
        self.B = system_params.B
        self.H = system_params.H
        self.Q = system_params.Q
        self.R = system_params.R

        self.x = initial_state.x
        self.Sigma = initial_state.Sigma

    def prediction(self, u=None):
        self.x_pred = self.A @ self.x
        if u is not None:
            self.x_pred += self.B @ u

        self.Sigma_pred = self.A @ self.Sigma @ self.A.T + self.Q

        self.z_hat = self.H @ self.x_pred

    def correction(self, z):
        self.v = z - self.z_hat

        # Innovation
        self.S = self.H @ self.Sigma_pred @ self.H.T + self.R

        # Kalman Gain
        self.K = self.Sigma_pred @ self.H.T @ np.linalg.inv(self.S)

        # Correction
        self.x = self.x_pred + self.K @ self.v

        I = np.eye(self.x.shape[0])

        self.Sigma = (I - self.K @ self.H) @ self.Sigma_pred @ \
                        (I - self.K @ self.H).T + \
                        self.K @ self.R @ self.K.T
