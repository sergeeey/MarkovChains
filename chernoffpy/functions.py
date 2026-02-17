"""
Chernoff functions: operator-valued functions C(t) satisfying Chernoff's conditions.

Each ChernoffFunction acts on functions f: R -> R (or R -> C) and returns a new function.
The n-fold composition C(t/n)^n f approximates e^{tA} f (the semigroup).

Terminology:
- "order k" means C^{(j)}(0) = A^j for j = 0, 1, ..., k
- By Galkin-Remizov (2025): order k => convergence rate O(1/n^k)

Implemented:
1. PhysicalG(t)   — 1st-order, weighted average with shifts ±2√t (from arXiv:2301.05284)
2. PhysicalS(t)   — 2nd-order, weighted average with shifts ±√(6t) (from arXiv:2301.05284)
3. BackwardEuler  — 1st-order, Fourier multiplier 1/(1+tξ²)
4. CrankNicolson  — 2nd-order, Fourier multiplier (1-tξ²/2)/(1+tξ²/2)
5. PadeChernoff   — arbitrary order via Padé approximants of e^z
"""

import numpy as np
from abc import ABC, abstractmethod


class ChernoffFunction(ABC):
    """Abstract base class for Chernoff functions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""

    @property
    @abstractmethod
    def order(self) -> int:
        """Order of tangency k (number of matching derivatives)."""

    @abstractmethod
    def apply(self, f_values: np.ndarray, x_grid: np.ndarray, t: float) -> np.ndarray:
        """Apply C(t) to discretized function values on x_grid."""

    def compose(self, f_values: np.ndarray, x_grid: np.ndarray, t: float, n: int) -> np.ndarray:
        """Compute C(t/n)^n f — the Chernoff approximation with n steps.

        This is the core computation: iteratively apply C(t/n) n times.
        By Chernoff's theorem: C(t/n)^n f -> e^{tA} f as n -> infinity.
        By Galkin-Remizov: error = O(1/n^k) where k = self.order.
        """
        result = f_values.copy()
        dt = t / n
        for _ in range(n):
            result = self.apply(result, x_grid, dt)
        return result

    def compose_all(self, f_values: np.ndarray, x_grid: np.ndarray, t: float,
                    n_max: int) -> list[np.ndarray]:
        """Compute C(t/n)^n f for n = 1, 2, ..., n_max. Returns list of arrays."""
        results = []
        for n in range(1, n_max + 1):
            results.append(self.compose(f_values, x_grid, t, n))
        return results


# ---------------------------------------------------------------------------
# Physical-space Chernoff functions (from arXiv:2301.05284)
# These operate by evaluating f at shifted points — no Fourier transform needed
# ---------------------------------------------------------------------------

class PhysicalG(ChernoffFunction):
    """First-order Chernoff function from Dragunova-Nikbakht-Remizov (2023).

    (G(t)f)(x) = (1/2)f(x) + (1/4)f(x + 2√t) + (1/4)f(x - 2√t)

    Order: k = 1 (first-order tangency to the Laplacian d²/dx²)
    Convergence rate: O(1/n) for f ∈ H²
    """

    @property
    def name(self) -> str:
        return "G(t) [order 1, physical]"

    @property
    def order(self) -> int:
        return 1

    def apply(self, f_values: np.ndarray, x_grid: np.ndarray, t: float) -> np.ndarray:
        if t == 0:
            return f_values.copy()
        shift = 2 * np.sqrt(t)
        dx = x_grid[1] - x_grid[0]
        shift_idx = shift / dx

        f_plus = np.interp(x_grid + shift, x_grid, f_values, period=x_grid[-1] - x_grid[0] + dx)
        f_minus = np.interp(x_grid - shift, x_grid, f_values, period=x_grid[-1] - x_grid[0] + dx)

        return 0.5 * f_values + 0.25 * f_plus + 0.25 * f_minus


class PhysicalS(ChernoffFunction):
    """Second-order Chernoff function from Dragunova-Nikbakht-Remizov (2023).

    (S(t)f)(x) = (2/3)f(x) + (1/6)f(x + √(6t)) + (1/6)f(x - √(6t))

    Order: k = 2 (second-order tangency to the Laplacian d²/dx²)
    Convergence rate: O(1/n²) for f ∈ H⁴
    """

    @property
    def name(self) -> str:
        return "S(t) [order 2, physical]"

    @property
    def order(self) -> int:
        return 2

    def apply(self, f_values: np.ndarray, x_grid: np.ndarray, t: float) -> np.ndarray:
        if t == 0:
            return f_values.copy()
        shift = np.sqrt(6 * t)
        dx = x_grid[1] - x_grid[0]

        f_plus = np.interp(x_grid + shift, x_grid, f_values, period=x_grid[-1] - x_grid[0] + dx)
        f_minus = np.interp(x_grid - shift, x_grid, f_values, period=x_grid[-1] - x_grid[0] + dx)

        return (2/3) * f_values + (1/6) * f_plus + (1/6) * f_minus


# ---------------------------------------------------------------------------
# Fourier-space Chernoff functions
# These operate via multipliers in Fourier space — efficient for uniform grids
# ---------------------------------------------------------------------------

class _FourierChernoff(ChernoffFunction):
    """Base class for Chernoff functions defined via Fourier multipliers.

    For A = d²/dx² on L²(R), the Fourier symbol of A is -ξ².
    A Chernoff function C(t) acts in Fourier space as multiplication by r(-tξ²),
    where r(z) is a rational approximation to e^z.
    """

    @abstractmethod
    def multiplier(self, xi_sq: np.ndarray, t: float) -> np.ndarray:
        """Fourier multiplier r(-t*xi²) for each frequency."""

    def apply(self, f_values: np.ndarray, x_grid: np.ndarray, t: float) -> np.ndarray:
        if t == 0:
            return f_values.copy()
        N = len(f_values)
        f_hat = np.fft.rfft(f_values)
        dx = x_grid[1] - x_grid[0]
        freqs = np.fft.rfftfreq(N, d=dx) * 2 * np.pi  # angular frequencies
        xi_sq = freqs ** 2
        mult = self.multiplier(xi_sq, t)
        return np.fft.irfft(f_hat * mult, n=N)


class BackwardEuler(_FourierChernoff):
    """Backward Euler (implicit Euler) Chernoff function.

    C(t) = (I - tA)^{-1}
    Fourier multiplier: 1 / (1 + tξ²)

    Order: k = 1
    Padé equivalent: [0/1] approximant of e^z
    Convergence rate: O(1/n) for f ∈ H²
    """

    @property
    def name(self) -> str:
        return "Backward Euler [order 1, Padé [0/1]]"

    @property
    def order(self) -> int:
        return 1

    def multiplier(self, xi_sq: np.ndarray, t: float) -> np.ndarray:
        return 1.0 / (1.0 + t * xi_sq)


class CrankNicolson(_FourierChernoff):
    """Crank-Nicolson (Cayley transform) Chernoff function.

    C(t) = (I + tA/2)(I - tA/2)^{-1}
    Fourier multiplier: (1 - tξ²/2) / (1 + tξ²/2)

    Order: k = 2
    Padé equivalent: [1/1] approximant of e^z
    Convergence rate: O(1/n²) for f ∈ H⁴
    """

    @property
    def name(self) -> str:
        return "Crank-Nicolson [order 2, Padé [1/1]]"

    @property
    def order(self) -> int:
        return 2

    def multiplier(self, xi_sq: np.ndarray, t: float) -> np.ndarray:
        u = t * xi_sq
        return (1.0 - u / 2) / (1.0 + u / 2)


class PadeChernoff(_FourierChernoff):
    """Chernoff function from Padé [m/n] approximant of e^z.

    Padé [m/n] matches e^z to order m+n (i.e., k = m+n matching derivatives).
    For the heat equation (A = d²/dx²): z = -tξ².

    Order: k = m + n
    Convergence rate: O(1/n^{m+n}) for f ∈ H^{2(m+n)}

    A-stability (|R(z)| ≤ 1 for Re(z) ≤ 0) requires m ≤ n.

    Predefined:
    - PadeChernoff(0, 1) = Backward Euler
    - PadeChernoff(1, 1) = Crank-Nicolson
    - PadeChernoff(2, 1) = 3rd-order scheme
    - PadeChernoff(1, 2) = 3rd-order A-stable scheme
    """

    def __init__(self, m: int = 2, n: int = 1):
        """Create Padé [m/n] Chernoff function.

        Args:
            m: degree of numerator polynomial
            n: degree of denominator polynomial
        """
        self.m = m
        self.n = n
        self.a_stable = m <= n
        self._p_coeffs, self._q_coeffs = _compute_pade_coefficients(m, n)
        if not self.a_stable:
            import warnings
            warnings.warn(
                f"Padé [{m}/{n}] is NOT A-stable (m > n). "
                f"High-frequency components will be amplified, causing divergence. "
                f"Use Padé [{n}/{m}] or [{m-1}/{n+1}] instead for stability."
            )

    @property
    def name(self) -> str:
        return f"Padé [{self.m}/{self.n}] [order {self.order}]"

    @property
    def order(self) -> int:
        return self.m + self.n

    def multiplier(self, xi_sq: np.ndarray, t: float) -> np.ndarray:
        z = -t * xi_sq  # z = tA in Fourier space (A -> -xi²)
        p = np.polyval(self._p_coeffs[::-1], z)  # numerator
        q = np.polyval(self._q_coeffs[::-1], z)  # denominator
        return p / q


def _compute_pade_coefficients(m: int, n: int) -> tuple[list[float], list[float]]:
    """Compute Padé [m/n] coefficients for e^z.

    Returns (p_coeffs, q_coeffs) where:
    - p_coeffs = [p_0, p_1, ..., p_m] (numerator)
    - q_coeffs = [q_0, q_1, ..., q_n] (denominator)
    - p(z)/q(z) = e^z + O(z^{m+n+1})

    Uses the explicit formula:
    p_j = C(m+n-j, m+n) * C(m, j) * m! / (m+n)!  ... actually use the standard formula:

    p_j = (m+n-j)! * m! / ((m+n)! * j! * (m-j)!)  for j = 0, ..., m
    q_j = (m+n-j)! * n! / ((m+n)! * j! * (n-j)!) * (-1)^j  for j = 0, ..., n
    """
    from math import factorial, comb

    p = []
    for j in range(m + 1):
        p_j = factorial(m + n - j) * factorial(m) / (factorial(m + n) * factorial(j) * factorial(m - j))
        p.append(p_j)

    q = []
    for j in range(n + 1):
        q_j = factorial(m + n - j) * factorial(n) / (factorial(m + n) * factorial(j) * factorial(n - j))
        q_j *= (-1) ** j
        q.append(q_j)

    return p, q
