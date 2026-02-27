# Mathematical Background

## The Chernoff product formula

The core idea behind ChernoffPy is the **Chernoff product formula** (Chernoff, 1968):

$$e^{tL} f = \lim_{n \to \infty} F\!\left(\tfrac{t}{n}\right)^n f$$

where:

- $L$ is a linear operator (the generator of a semigroup)
- $e^{tL}$ is the exact solution operator (the semigroup itself)
- $F(s)$ is the **Chernoff function** — a family of operators satisfying $F(0) = I$ and $F'(0) = L$

In practice we compute $F(t/n)^n f$ for a finite $n$, which approximates $e^{tL} f$ with error $O(1/n^k)$, where $k$ depends on how many derivatives of $F$ match those of $e^{tL}$ at $s=0$.

---

## Connection to option pricing

### Black-Scholes PDE

Under the risk-neutral measure, the value $V(S, t)$ of a European option satisfies:

$$\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0$$

### Wilmott substitution (BS → heat equation)

The substitution

$$x = \ln(S/K), \quad \tau = \tfrac{\sigma^2}{2}(T - t), \quad V(S, t) = K e^{\alpha x + \beta \tau} u(x, \tau)$$

with $\alpha = -(k-1)/2$, $\beta = -(k+1)^2/4$, $k = 2r/\sigma^2$, transforms the BS PDE into the **heat equation**:

$$\frac{\partial u}{\partial \tau} = \frac{\partial^2 u}{\partial x^2}$$

### Chernoff approximation

The heat equation generator is $L = \partial^2/\partial x^2$. We approximate $e^{\tau L}$ via:

$$u(\cdot, \tau) \approx F(\tau/n)^n\, u_0$$

Different choices of $F$ give different numerical schemes:

| Chernoff function $F(s)$ | Scheme | Order |
|--------------------------|--------|-------|
| $(I - s\partial^2_x)^{-1}$ | Backward Euler | 1 |
| $(I - \tfrac{s}{2}\partial^2_x)^{-1}(I + \tfrac{s}{2}\partial^2_x)$ | Crank-Nicolson | 2 |
| Padé[1/2] rational approx. | Padé-Chernoff | 3 |
| Padé[2/2] rational approx. | Padé-Chernoff | 4 |
| Gaussian convolution kernel | Physical-G | 1 |
| Sinc-kernel convolution | Physical-S | 2 |

---

## Certified error bounds

The key result from **Galkin & Remizov (2025)** is a rigorous upper bound on the approximation error:

$$\left\| e^{tL} f - F(t/n)^n f \right\| \leq C(f, L, F) \cdot \frac{1}{n^k}$$

where the constant $C$ depends on the regularity of the payoff $f$ and the specific Chernoff function $F$.

ChernoffPy computes this constant numerically and returns it as the `certified_bound`, giving a **provable** (not just empirical) accuracy guarantee.

---

## Barrier options: DST vs FFT

For barrier options, the boundary condition $V(B, t) = 0$ becomes $u(x_B, \tau) = 0$ after the Wilmott substitution. Two approaches are used:

**DST method (`BarrierDSTPricer`):**
Expand $u$ in a sine series that automatically satisfies Dirichlet BCs. No Gibbs oscillations. Slightly slower due to exact boundary enforcement.

**FFT method (`BarrierPricer`):**
Use fast Fourier convolution with a projection step at each time step. Faster but can exhibit Gibbs artifacts near the barrier.

---

## References

1. **Chernoff, P. R.** (1968). Note on product formulas for operator semigroups. *Journal of Functional Analysis*, 2(2), 238–242.

2. **Galkin, O. & Remizov, I.** (2025). Rate of convergence of Chernoff approximations for contractive Chernoff functions of the Schrödinger semigroup. *Israel Journal of Mathematics*. [DOI: 10.1007/s11856-024-2699-7](https://doi.org/10.1007/s11856-024-2699-7)

3. **Butko, Ya. A.** (2020). The method of Chernoff approximation. In *Semigroups of Operators – Theory and Applications*, Lecture Notes in Mathematics. Springer. [arXiv:1905.07309](https://arxiv.org/abs/1905.07309)

4. **Draganova, N. & Nikbakht, M.** (2023). Numerical realization of Chernoff product formulas for heat equation. [arXiv:2301.05284](https://arxiv.org/abs/2301.05284)
