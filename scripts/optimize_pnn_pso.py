"""Particle Swarm Optimization (PSO) for bounded continuous parameters."""

from __future__ import annotations

import numpy as np


def particle_swarm_optimize(
    objective,
    bounds,
    *,
    swarm_size=40,
    max_iterations=60,
    inertia=0.72,
    cognitive=1.49,
    social=1.49,
    periodic_dimensions=(),
    seed=42,
):
    """Minimize ``objective`` using PSO within per-coordinate bounds.

    Periodic coordinates (such as phase) wrap at their interval boundaries.
    Returns the best position, its objective value, and global-best history,
    including the initial swarm as iteration zero.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] == 0:
        raise ValueError("bounds must have shape (n_dimensions, 2).")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Each bound must be finite, with its maximum greater than its minimum.")
    if isinstance(swarm_size, bool) or not isinstance(swarm_size, (int, np.integer)) or swarm_size < 2:
        raise ValueError("swarm_size must be an integer greater than or equal to 2.")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if not all(np.isfinite(v) and v >= 0 for v in (inertia, cognitive, social)):
        raise ValueError("PSO coefficients must be finite and nonnegative.")

    n_dimensions = bounds.shape[0]
    lower, upper = bounds[:, 0], bounds[:, 1]
    span = upper - lower
    periodic = np.asarray(tuple(periodic_dimensions), dtype=int)
    if periodic.size and (np.any(periodic < 0) or np.any(periodic >= n_dimensions)):
        raise ValueError("periodic_dimensions contains an index outside bounds.")
    periodic_mask = np.zeros(n_dimensions, dtype=bool)
    periodic_mask[periodic] = True

    rng = np.random.default_rng(int(seed))
    positions = rng.uniform(lower, upper, size=(swarm_size, n_dimensions))
    velocities = rng.uniform(-0.1 * span, 0.1 * span, size=(swarm_size, n_dimensions))
    personal_positions = positions.copy()
    personal_values = np.asarray([objective(x.copy()) for x in positions], dtype=float)
    if not np.all(np.isfinite(personal_values)):
        raise ValueError("objective must return finite scalar values.")
    evaluations = swarm_size

    best_index = int(np.argmin(personal_values))
    global_position = personal_positions[best_index].copy()
    global_value = float(personal_values[best_index])
    history = [global_value]

    for _ in range(max_iterations):
        r1 = rng.random((swarm_size, n_dimensions))
        r2 = rng.random((swarm_size, n_dimensions))
        personal_delta = personal_positions - positions
        global_delta = global_position - positions
        if periodic_mask.any():
            p = periodic_mask
            personal_delta[:, p] = (personal_delta[:, p] + span[p] / 2) % span[p] - span[p] / 2
            global_delta[:, p] = (global_delta[:, p] + span[p] / 2) % span[p] - span[p] / 2

        velocities = (
            inertia * velocities
            + cognitive * r1 * personal_delta
            + social * r2 * global_delta
        )
        positions += velocities
        positions[:, ~periodic_mask] = np.clip(
            positions[:, ~periodic_mask], lower[~periodic_mask], upper[~periodic_mask]
        )
        if periodic_mask.any():
            positions[:, periodic_mask] = (
                (positions[:, periodic_mask] - lower[periodic_mask])
                % span[periodic_mask]
                + lower[periodic_mask]
            )

        values = np.asarray([objective(x.copy()) for x in positions], dtype=float)
        evaluations += swarm_size
        if not np.all(np.isfinite(values)):
            raise ValueError("objective must return finite scalar values.")
        improved = values < personal_values
        personal_positions[improved] = positions[improved]
        personal_values[improved] = values[improved]

        best_index = int(np.argmin(personal_values))
        if personal_values[best_index] < global_value:
            global_value = float(personal_values[best_index])
            global_position = personal_positions[best_index].copy()
        history.append(global_value)

    n_channels = n_dimensions // 2
    return {
        "position": global_position,
        "amplitude_weights": global_position[:n_channels].copy(),
        "phase_weights": global_position[n_channels:].copy(),
        "best_fitness": global_value,
        "best_fitness_history": np.asarray(history),
        "evaluations": evaluations,
        "iterations": max_iterations,
        "swarm_size": swarm_size,
    }
