"""
Exact semigroup solutions for comparison with Chernoff approximations.

Each semigroup computes e^{tA} f exactly (or to machine precision)
for a specific generator A on a specific space.
"""

import numpy as np
from scipy.integrate import quad


class HeatSemigroup:
    """Exact solution of the heat equation du/dt = d²u/dx² on R.

    The solution is given by convolution with the Gaussian kernel:
    (e^{tA} f)(x) = (1/√(4πt)) ∫ f(y) exp(-(x-y)²/(4t)) dy

    For periodic functions on [-π, π], the heat kernel becomes a theta function,
    but for practical purposes we use direct numerical integration or Fourier.
    """

    @staticmethod
    def solve_fourier(f_values: np.ndarray, x_grid: np.ndarray, t: float) -> np.ndarray:
        """Exact solution via Fourier transform (fast, for uniform grids).

        In Fourier space: (e^{tA} f)^(ξ) = e^{-tξ²} f^(ξ)
        """
        if t == 0:
            return f_values.copy()
        N = len(f_values)
        f_hat = np.fft.rfft(f_values)
        dx = x_grid[1] - x_grid[0]
        freqs = np.fft.rfftfreq(N, d=dx) * 2 * np.pi
        xi_sq = freqs ** 2
        return np.fft.irfft(f_hat * np.exp(-t * xi_sq), n=N)

    @staticmethod
    def solve_convolution(f_callable, x_grid: np.ndarray, t: float,
                          integration_limit: float = 20.0) -> np.ndarray:
        """Exact solution via numerical convolution with Gaussian kernel.

        More accurate for non-periodic or irregular functions, but slower.

        Args:
            f_callable: initial condition as a callable f(x)
            x_grid: spatial grid points
            t: time
            integration_limit: limits of integration [-L, L]
        """
        if t == 0:
            return np.array([f_callable(x) for x in x_grid])

        kernel_factor = 1.0 / np.sqrt(4 * np.pi * t)
        result = np.zeros_like(x_grid, dtype=float)

        for i, x in enumerate(x_grid):
            def integrand(y):
                return f_callable(y) * np.exp(-(x - y)**2 / (4 * t))
            val, _ = quad(integrand, -integration_limit, integration_limit, limit=100)
            result[i] = kernel_factor * val

        return result

    @staticmethod
    def eigenfunction_solution(x_grid: np.ndarray, t: float,
                               coeffs: np.ndarray, L: float = 1.0) -> np.ndarray:
        """Exact solution on [0, L] with Dirichlet BCs via eigenfunction expansion.

        u(t,x) = Σ c_n e^{-n²π²t/L²} √(2/L) sin(nπx/L)

        This is the method from Problems 3-4 of our self-test.

        Args:
            x_grid: spatial grid on [0, L]
            coeffs: Fourier sine coefficients c_n (n = 1, 2, ...)
            t: time
            L: domain length
        """
        result = np.zeros_like(x_grid, dtype=float)
        norm_factor = np.sqrt(2.0 / L)
        for n_idx, c_n in enumerate(coeffs):
            n = n_idx + 1
            eigenvalue = -(n * np.pi / L) ** 2
            result += c_n * np.exp(eigenvalue * t) * norm_factor * np.sin(n * np.pi * x_grid / L)
        return result
