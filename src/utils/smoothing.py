import numpy as np


# Exponential moving average filter for 2D points
class ExponentialMovingAverageSmoother:

    # Initialize filter with smoothing factor alpha in (0, 1]
    def __init__(self, alpha: float = 0.6) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: tuple[float, float] | None = None

    # Update smoother with new point and return smoothed coordinates
    def update(self, x: float, y: float) -> tuple[float, float]:
        if self._state is None:
            self._state = (x, y)
            return x, y
        sx = self.alpha * x + (1 - self.alpha) * self._state[0]
        sy = self.alpha * y + (1 - self.alpha) * self._state[1]
        self._state = (sx, sy)
        return sx, sy

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        return self.update(x, y)


class KalmanSmoother:

    def __init__(self, process_variance: float = 1e-2, measurement_variance: float = 1.0) -> None:
        self.dt = 1.0
        self._a = np.array([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float)
        self._h = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=float)
        self._q = np.eye(4) * process_variance
        self._r = np.eye(2) * measurement_variance
        self._x: np.ndarray | None = None
        self._p = np.eye(4)

    def update(self, x: float, y: float) -> tuple[float, float]:
        z = np.array([x, y], dtype=float)
        if self._x is None:
            self._x = np.array([x, y, 0.0, 0.0], dtype=float)
            return float(x), float(y)

        self._x = self._a @ self._x
        self._p = self._a @ self._p @ self._a.T + self._q

        S = self._h @ self._p @ self._h.T + self._r
        K = self._p @ self._h.T @ np.linalg.inv(S)
        y_residual = z - (self._h @ self._x)
        self._x = self._x + K @ y_residual
        self._p = (np.eye(4) - K @ self._h) @ self._p

        return float(self._x[0]), float(self._x[1])

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        return self.update(x, y)
