# Core: Chernoff Functions

The core module implements the abstract `ChernoffFunction` base class and five concrete schemes.

## Abstract base

::: chernoffpy.functions.ChernoffFunction
    options:
      members:
        - apply
        - compose

## Schemes

::: chernoffpy.functions.BackwardEuler

::: chernoffpy.functions.CrankNicolson

::: chernoffpy.functions.PadeChernoff

::: chernoffpy.functions.PhysicalG

::: chernoffpy.functions.PhysicalS

## Heat semigroup (exact reference)

::: chernoffpy.semigroups.HeatSemigroup

## Analysis utilities

::: chernoffpy.analysis.compute_errors

::: chernoffpy.analysis.convergence_rate

::: chernoffpy.analysis.convergence_table

## Backend abstraction

::: chernoffpy.backends.get_backend

::: chernoffpy.backends.to_backend

::: chernoffpy.backends.to_numpy
