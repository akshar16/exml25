"""Training runner for Q-learning using the environment.

This runner wires the F1Game environment to the Q-learning trainer with
lightweight reset/step callbacks. It does NOT render for speed.
"""

import os
import sys
from pathlib import Path

# Ensure repository root on sys.path for absolute imports
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.game import F1Game
from env.controls import forward, back, steer_right, steer_left, brake, boost, reset
from models.my_model.model import train_q_learning


_GAME: F1Game = None
_CAR_IDX: int = None
_PREV_PROGRESS: float = 0.0
_PREV_COLLIDED: bool = False


def _find_self_index(models_dir: str = 'models', self_name: str = 'my_model') -> int:
    names = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    for i, name in enumerate(names):
        if name == self_name:
            return i
    return 0


def env_reset_fn():
    global _GAME, _CAR_IDX, _PREV_PROGRESS, _PREV_COLLIDED
    _GAME = F1Game()
    _CAR_IDX = _find_self_index()
    car = _GAME._cars[_CAR_IDX]
    reset(car)
    obs = car.get_observation()
    _PREV_PROGRESS = float(obs.get('lap_progress', 0.0))
    _PREV_COLLIDED = False
    return obs, car


def _apply_action_to_car(car, action):
    thr = float(action.get('throttle', 0.0))
    steer = float(action.get('steer', 0.0))
    do_brake = bool(action.get('brake', False))

    # Map discrete action to env controls
    if thr > 0.5:
        forward(car)
    elif thr < -0.5:
        back(car)

    if steer > 0.5:
        steer_right(car)
    elif steer < -0.5:
        steer_left(car)

    if do_brake:
        brake(car)


def env_step_fn(action):
    global _GAME, _CAR_IDX, _PREV_PROGRESS, _PREV_COLLIDED
    car = _GAME._cars[_CAR_IDX]

    # Apply action
    _apply_action_to_car(car, action)

    # Handle pygame events to keep window responsive
    _GAME.handle_events()

    # Step the environment
    _GAME.step()
    
    # Render to show the car racing
    _GAME.render()

    obs = car.get_observation()
    progress = float(obs.get('lap_progress', 0.0))
    speed = float(obs.get('speed', 0.0))
    collided = bool(obs.get('collided', False))

    # Reward shaping (inspired by Code Bullet's gates)
    dprog = max(0.0, progress - _PREV_PROGRESS)
    reward = 50.0 * dprog + 0.05 * speed

    # Strong collision penalties
    if collided:
        if not _PREV_COLLIDED:
            reward -= 50.0  # initial collision penalty (strong)
        else:
            reward -= 10.0  # sustained collision penalty (every step)
    
    _PREV_PROGRESS = progress
    _PREV_COLLIDED = collided

    done = False
    if progress >= 0.99:
        reward += 100.0  # goal reward for completing lap
        done = True

    return obs, reward, done, car


def main():
    train_q_learning(
        env_reset_fn=env_reset_fn,
        env_step_fn=env_step_fn,
        episodes=100,  # Start with 100 episodes
        max_steps=2000,
        gamma=0.99,
        lr=1e-3,
        batch_size=32,
        target_update_freq=5,
    )


if __name__ == "__main__":
    main()


import os
import sys
from pathlib import Path
import time
from typing import Tuple, Dict, Any

import numpy as np

# Ensure repository root on sys.path for absolute imports
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.game import F1Game
from env.controls import forward, back, steer_right, steer_left, brake, boost, reset
from models.my_model.model import train_rl


# Globals for current episode/game state
_GAME: F1Game = None
_CAR_IDX: int = None
_PREV_PROGRESS: float = 0.0


def _find_self_index(models_dir: str = 'models', self_name: str = 'my_model') -> int:
    names = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    # Maintain same ordering as env.game uses
    for i, name in enumerate(names):
        if name == self_name:
            return i
    # Fallback: if not found, default to 0
    return 0


def env_reset_fn() -> Dict[str, Any]:
    global _GAME, _CAR_IDX, _PREV_PROGRESS
    _GAME = F1Game()
    _CAR_IDX = _find_self_index()
    car = _GAME._cars[_CAR_IDX]
    reset(car)
    obs = car.get_observation()
    _PREV_PROGRESS = float(obs.get('lap_progress', 0.0))
    return obs


def _apply_action_to_car(car, action: Dict[str, float]):
    thr = float(action.get('throttle', 0.0))
    steer = float(action.get('steering', 0.0))
    do_brake = int(action.get('brake', 0))
    do_boost = int(action.get('boost', 0))

    # Map continuous to discrete controls with thresholds
    if thr > 0.2:
        forward(car)
    elif thr < -0.2:
        back(car)

    if steer > 0.2:
        steer_right(car)
    elif steer < -0.2:
        steer_left(car)

    if do_brake:
        brake(car)
    if do_boost:
        try:
            boost(car)
        except Exception:
            pass


def env_step_fn(action: Dict[str, float]) -> Tuple[Dict[str, Any], float, bool]:
    global _GAME, _CAR_IDX, _PREV_PROGRESS
    car = _GAME._cars[_CAR_IDX]

    # Apply action
    _apply_action_to_car(car, action)

    # Advance the world one step without rendering
    _GAME.step()

    obs = car.get_observation()
    progress = float(obs.get('lap_progress', 0.0))
    speed = float(obs.get('speed', 0.0))
    collided = bool(obs.get('collided', False))

    # Reward shaping
    dprog = max(0.0, progress - _PREV_PROGRESS)
    _PREV_PROGRESS = progress
    reward = 100.0 * dprog + 0.1 * speed
    if collided:
        reward -= 1.0

    done = False
    # Consider finishing a lap as episode termination
    if progress >= 0.99:
        reward += 50.0
        done = True

    return obs, reward, done


def main():
    # Train for a modest number of episodes by default
    train_rl(
        env_reset_fn=env_reset_fn,
        env_step_fn=env_step_fn,
        episodes=20,
        max_steps=1500,
        gamma=0.99,
        lr=1e-3,
        std=0.2,
    )


if __name__ == "__main__":
    main()
