"""Covariance estimation from individual unforced trajectories.

The returned matrix is the covariance of a complete observable time series
from one trajectory.  For an ensemble mean over ``n_ens`` independent
trajectories, its covariance is this matrix divided by ``n_ens``.  That scalar
factor cancels from a GLS point estimate and is therefore not applied here.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np


ArrayFunction = Callable[[np.ndarray], np.ndarray]


def shrink_covariance(covariance: np.ndarray, alpha: float = 1.0e-2) -> np.ndarray:
    """Apply isotropic shrinkage while preserving the average variance."""

    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    covariance = 0.5 * (covariance + covariance.T)
    dimension = covariance.shape[0]
    target_variance = np.trace(covariance) / dimension
    shrunk = (1.0 - alpha) * covariance + alpha * target_variance * np.eye(
        dimension
    )
    return 0.5 * (shrunk + shrunk.T)


def estimate_control_covariance(
    initial_states: np.ndarray,
    *,
    drift: ArrayFunction,
    observables: Sequence[ArrayFunction],
    n_steps: int,
    dt: float,
    sigma: float,
    rng: np.random.Generator,
    alpha: float = 1.0e-2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate covariance across stacked individual control trajectories.

    Row ``i`` of the returned trajectory matrix contains all retained times of
    the first observable for trajectory ``i``, followed by all retained times
    of the second observable, and so on.  Time points are not treated as
    independent covariance samples.
    """

    states = np.asarray(initial_states, dtype=np.float64).copy()
    if states.ndim != 1 or states.size < 2:
        raise ValueError("initial_states must contain at least two trajectories")
    if n_steps < 1 or dt <= 0.0 or sigma < 0.0:
        raise ValueError("n_steps and dt must be positive and sigma non-negative")
    if not observables:
        raise ValueError("at least one observable is required")

    n_trajectories = states.size
    n_observables = len(observables)
    trajectory_vectors = np.empty(
        (n_trajectories, n_observables * n_steps), dtype=np.float64
    )
    noise_scale = sigma * np.sqrt(dt)

    for time_index in range(n_steps):
        states += drift(states) * dt + noise_scale * rng.standard_normal(
            n_trajectories
        )
        for observable_index, observable in enumerate(observables):
            values = np.asarray(observable(states), dtype=np.float64)
            if values.shape != states.shape:
                raise ValueError("each observable must return one value per trajectory")
            trajectory_vectors[
                :, observable_index * n_steps + time_index
            ] = values

    empirical = np.cov(trajectory_vectors, rowvar=False, ddof=1)
    empirical = np.atleast_2d(empirical)
    covariance = shrink_covariance(empirical, alpha=alpha)
    return covariance, empirical, trajectory_vectors
