"""Small dependency-free offline models for the first GPU-LJ study.

The models intentionally use only NumPy: this host has neither PyTorch nor
scikit-learn.  `fit_fqi` is batch fitted Q-iteration over the full hybrid
delta action space; it is the sequential offline-RL comparator.  `gamma=0`
is the matched myopic contextual-runtime model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SKIN_MIN, SKIN_MAX = 0.6, 1.2
EVERY_MIN, EVERY_MAX = 1, 10
EVERY_DELTAS = (-5, -1, 0, 1, 5)


def _number(value: Any) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def state_vector(obs: dict[str, Any]) -> np.ndarray:
    """Observable-only state.  Deliberately excludes contention labels."""
    previous = obs.get("previous", {})
    timing = previous.get("lammps", {}).get("timing_avg_seconds", {})
    gpu = obs.get("gpu", {})
    values = [
        obs.get("skin"), obs.get("every"),
        previous.get("wall_seconds"), previous.get("throughput"),
        previous.get("temperature"), previous.get("pe"),
        previous.get("neighbor_builds"), previous.get("dangerous_builds"),
        timing.get("pair"), timing.get("neigh"), timing.get("comm"),
        timing.get("modify"), timing.get("other"),
        gpu.get("temperature"), gpu.get("power"), gpu.get("sm_clock"),
        gpu.get("mem_clock"), gpu.get("util"), gpu.get("mem_util"),
        gpu.get("mem_used"),
    ]
    return np.array([_number(x) for x in values], dtype=float)


def valid_actions(obs: dict[str, Any], skin_points: int = 21) -> list[tuple[float, int]]:
    skin = _number(obs.get("skin"))
    every = int(round(_number(obs.get("every"))))
    skin_deltas = np.linspace(max(-0.2, SKIN_MIN - skin), min(0.2, SKIN_MAX - skin), skin_points)
    return [(float(delta), de) for de in EVERY_DELTAS if EVERY_MIN <= every + de <= EVERY_MAX for delta in skin_deltas]


def action_vector(action: tuple[float, int]) -> np.ndarray:
    ds, de = action
    onehot = [float(de == branch) for branch in EVERY_DELTAS]
    return np.array([float(ds) / 0.2, *onehot], dtype=float)


def feature_matrix(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Linear action effects plus state-action interactions, with intercept."""
    interaction = (states[:, :, None] * actions[:, None, :]).reshape(len(states), -1)
    return np.concatenate((np.ones((len(states), 1)), states, actions, interaction), axis=1)


class LinearQ:
    def __init__(self, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray):
        self.mean, self.scale, self.coef = mean, scale, coef

    @classmethod
    def fit(cls, states: np.ndarray, actions: np.ndarray, targets: np.ndarray, ridge: float = 1e-2) -> "LinearQ":
        mean = states.mean(axis=0)
        scale = states.std(axis=0)
        scale[scale < 1e-8] = 1.0
        xs = (states - mean) / scale
        features = feature_matrix(xs, actions)
        penalty = np.eye(features.shape[1]) * ridge
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(features.T @ features + penalty, features.T @ targets)
        return cls(mean, scale, coef)

    def predict(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return feature_matrix((states - self.mean) / self.scale, actions) @ self.coef

    def best_action(self, obs: dict[str, Any]) -> tuple[float, int, float]:
        state = state_vector(obs)[None, :]
        choices = valid_actions(obs)
        acts = np.array([action_vector(a) for a in choices])
        scores = self.predict(np.repeat(state, len(choices), axis=0), acts)
        index = int(np.argmax(scores))
        return (*choices[index], float(scores[index]))

    def dump(self, path: Path, metadata: dict[str, Any]) -> None:
        path.write_text(json.dumps({"mean": self.mean.tolist(), "scale": self.scale.tolist(), "coef": self.coef.tolist(), **metadata}, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "LinearQ":
        data = json.loads(path.read_text())
        return cls(np.array(data["mean"]), np.array(data["scale"]), np.array(data["coef"]))


def rows_to_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.array([state_vector(row["state_before"]) for row in rows])
    next_states = np.array([state_vector(row["state_after"]) for row in rows])
    actions = np.array([action_vector(tuple(row["applied_action"])) for row in rows])
    rewards = np.array([float(row["reward"]) for row in rows])
    terminal = np.array([row["decision"] == 9 or bool(row["terminated"]) for row in rows])
    return states, actions, rewards, next_states, terminal


def fit_fqi(rows: list[dict[str, Any]], gamma: float, iterations: int = 30, ridge: float = 1e-2) -> tuple[LinearQ, list[float]]:
    states, actions, rewards, next_states, terminal = rows_to_arrays(rows)
    q = LinearQ.fit(states, actions, rewards, ridge)
    losses: list[float] = []
    for _ in range(iterations if gamma else 1):
        values = np.zeros(len(rows))
        for i, row in enumerate(rows):
            if terminal[i]:
                continue
            choices = valid_actions(row["state_after"])
            act = np.array([action_vector(x) for x in choices])
            ns = np.repeat(next_states[i:i+1], len(choices), axis=0)
            values[i] = float(q.predict(ns, act).max())
        target = rewards + gamma * values
        q = LinearQ.fit(states, actions, target, ridge)
        losses.append(float(np.mean((q.predict(states, actions) - target) ** 2)))
    return q, losses
