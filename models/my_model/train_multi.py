import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from env.game import F1Game
from env.controls import forward, back, steer_right, steer_left, brake, reset, boost
from models.my_model.model import _raycast_sensors


_GAME: F1Game = None
_NUM_CARS = 1  # Train 6 cars simultaneously per latest user request
_STUCK_THRESHOLD = 200  # Steps without progress before considering stuck (increased from 75)
_RENDER_EVERY = 4  # Render every N steps for speed (1=always, 4=every 4th frame)
_EVOLUTION_ENABLED = True  # Enable evolutionary selection of best performers
_BEST_PERFORMANCE_SCORE = 0.0  # Track best distance + survival for model saving


def env_reset_fn_multi():
    """Reset all training cars and return their initial states."""
    global _GAME
    _GAME = F1Game()
    cars = _GAME._cars[:_NUM_CARS]
    for car in cars:
        reset(car)
    
    # Reset episode rewards tracking
    env_step_fn_multi._episode_rewards = {i: 0.0 for i in range(_NUM_CARS)}
    if hasattr(_GAME, '_car_scores'):
        for i in range(_NUM_CARS):
            _GAME._car_scores[i] = 0.0
    
    result = {}
    for i, car in enumerate(cars):
        obs = car.get_observation()
        result[i] = {'obs': obs, 'car': car}
    
    return result


def env_step_fn_multi(actions_by_car):
    """Step all cars and return observations, rewards, dones, cars.
    
    Parameters:
    - actions_by_car: dict mapping car_idx -> action_dict
    
    Returns:
    - list of (obs, reward, done, car, car_idx) for each car
    """
    global _GAME
    
    # Apply all actions
    for car_idx, action in actions_by_car.items():
        car = _GAME._cars[car_idx]
        _apply_action_to_car(car, action)
    
    # Step environment once (all cars advance)
    _GAME.handle_events()
    _GAME.step()
    # Render less frequently for speed
    if not hasattr(env_step_fn_multi, '_step_count'):
        env_step_fn_multi._step_count = 0
    env_step_fn_multi._step_count += 1
    if env_step_fn_multi._step_count % _RENDER_EVERY == 0:
        _GAME.render()
    
    # Get results for each car
    results = []
    for car_idx in range(_NUM_CARS):
        car = _GAME._cars[car_idx]
        obs = car.get_observation()
        action_used = actions_by_car.get(car_idx)
        reward = _compute_reward(obs, car_idx, action_taken=action_used)
        done = float(obs.get('lap_progress', 0.0)) >= 0.99
        results.append((obs, reward, done, car, car_idx))
        
        # Update car score in game for display
        env_step_fn_multi._episode_rewards[car_idx] = env_step_fn_multi._episode_rewards.get(car_idx, 0.0) + reward
        if hasattr(_GAME, '_car_scores'):
            _GAME._car_scores[car_idx] = env_step_fn_multi._episode_rewards[car_idx]
    
    return results


def _apply_action_to_car(car, action):
    """Apply discrete action to car via environment controls."""
    thr = float(action.get('throttle', 0.0))
    steer = float(action.get('steer', 0.0))
    do_brake = bool(action.get('brake', False))
    do_boost = bool(action.get('boost', True))

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
    if do_boost:
        # Request boost only when moving roughly straight to avoid waste
        boost(car)


# Track state per car
_CAR_STATE = {}  # car_idx -> {'progress', 'collided', 'stuck_counter', 'last_progress', 'steps_alive', 'checkpoints_passed': int}
_CAR_PERFORMANCE = {}  # car_idx -> {'total_distance': float, 'avg_survival': float, 'episodes': int}


def _compute_reward(obs, car_idx, action_taken=None):
    """Compute reward for a single car - PRIORITIZE COLLISION AVOIDANCE FIRST.
    
    Args:
        obs: observation dict
        car_idx: index of the car
        action_taken: the action dict that was executed (for action-conditional rewards)
    """
    
    progress = float(obs.get('lap_progress', 0.0))
    speed = float(obs.get('speed', 0.0))
    collided = bool(obs.get('collided', False))
    all_coords = obs.get('all_coords', [])
    
    car = _GAME._cars[car_idx]  # Get car object first
    car_x, car_y = car.get_position()
    
    # Initialize tracking for new car
    if car_idx not in _CAR_STATE:
        _CAR_STATE[car_idx] = {
            'progress': 0.0,
            'collided': False,
            'stuck_counter': 0,
            'last_progress': 0.0,
            'steps_alive': 0,
            'last_pos': car.get_position(),
            'total_distance': 0.0,
            'checkpoints_passed': 0,
            'last_checkpoint_count': 0,
            'checkpoint_step_timer': 0,  # Tracks steps since last checkpoint
            'last_lap_count': 0,  # Track laps completed
            'lap_step_timer': 0  # Tracks steps for entire lap
        }
    
    prev_progress = _CAR_STATE[car_idx]['progress']
    prev_collided = _CAR_STATE[car_idx]['collided']
    last_prog = _CAR_STATE[car_idx]['last_progress']
    steps_alive = _CAR_STATE[car_idx].get('steps_alive', 0)
    last_pos = _CAR_STATE[car_idx].get('last_pos', (0, 0))

    # Acquire current position early (used for micro-progress regardless of other cars)
    car_x, car_y = car.get_position()

    # 9-ray expanded sensor layout: indices
    # 0:-135, 1:-90, 2:-45, 3:-27.5, 4:0, 5:27.5, 6:45, 7:90, 8:135
    rays = _raycast_sensors(car, num_rays=9, max_distance=300.0)
    forward_ray = rays[4]
    # Aggregate mid-left/right using the better (max) of the two near-forward angles
    left_mid = max(rays[2], rays[3])
    right_mid = max(rays[5], rays[6])
    # Side/extreme clearances
    left_side = rays[1]
    right_side = rays[7]
    left_diag = rays[0]
    right_diag = rays[8]

    # PHASE 1: SURVIVAL & COLLISION AVOIDANCE (most important)
    reward = 0.0
    
    # Survival bonus: reward for staying alive without crashing
    if not collided:
        reward += 1.0  # +1 per step alive (encourages careful driving)
        _CAR_STATE[car_idx]['steps_alive'] = steps_alive + 1
    else:
        _CAR_STATE[car_idx]['steps_alive'] = 0
    
    # Balanced collision penalties (learn to avoid walls without being overly conservative)
    if collided:
        if not prev_collided:
            reward -= 2000.0  # Heavy initial collision penalty
        else:
            reward -= 2000.0  # Sustained collision penalty

    # Ray-based clearance rewards encourage driving straight with space on both sides
    lateral_balance = 1.0 - abs(left_mid - right_mid)
    side_clearance = min(left_mid, right_mid)
    # Extreme/side clearances: use side + diagonal for broader spatial awareness
    side_clear = min(left_side, right_side)
    diag_clear = min(left_diag, right_diag)
    ray_reward = (
        6.0 * forward_ray +          # prioritize forward space
        3.0 * side_clear +           # maintain room on both sides
        1.5 * diag_clear +           # awareness of far diagonals
        2.0 * lateral_balance        # encourage centered driving
    )
    reward += ray_reward

    # CRITICAL OBSTACLE PROXIMITY PENALTIES: Punish getting too close BEFORE collision
    min_ray = min(rays)
    if min_ray < 0.15:  # Very close to obstacle
        reward -= 5.0  # Minimal penalty for danger zone (favor speed)
    elif min_ray < 0.25:  # Close to obstacle
        reward -= 1.0  # Gentle warning penalty
    elif min_ray < 0.35:  # Approaching obstacle
        reward -= 0.5  # Light warning only

    # STRONG PREVENTIVE REWARD: Big bonus for maintaining safe distance
    if forward_ray > 0.4 and min(rays) > 0.2:  # Safe clearance on all sides
        reward += 25.0  # Substantial safety bonus
    elif forward_ray > 0.3:
        reward += 10.0  # Moderate safety bonus

    # Encourage steering toward the most open direction (increased weight)
    longest_idx = int(np.argmax(rays))
    center_idx = 4  # forward ray index in 9-ray layout
    offset = abs(longest_idx - center_idx)
    reward += max(0.0, 20.0 - 8.0 * offset)  # 20pts aligned, 12pts one step, 4pts two steps
    
    # ACTION-ALIGNMENT BONUS: MASSIVELY REWARD turning toward longest ray (make it obvious!)
    if action_taken is not None:
        action_steer = float(action_taken.get('steer', 0.0))
        
        # Determine which direction the longest ray is in
        if longest_idx < center_idx:  # Longest ray is to the left
            desired_steer_direction = -1.0  # Should turn left
        elif longest_idx > center_idx:  # Longest ray is to the right
            desired_steer_direction = 1.0  # Should turn right
        else:  # Longest ray is forward
            desired_steer_direction = 0.0  # Should go straight
        
        # HUGE rewards for aligning with longest ray direction
        if desired_steer_direction == 0.0:
            # Longest ray is forward - reward going straight
            if abs(action_steer) < 0.3:
                reward += 50.0  # MASSIVE bonus for going straight when clear ahead
                
                # EXTRA BONUS: Encourage full throttle when path is straight and clear
                action_throttle = float(action_taken.get('throttle', 0.0))
                if action_throttle > 0.8:  # Full throttle
                    reward += 40.0  # Slightly bigger bonus for accelerating when clear (was 30)
                    # Additional speed-based bonus when going fast in the right direction
                    reward += 10.0 * speed  # Increased speed scaling (was 5.0)
            else:
                reward -= 20.0  # Penalty for turning when should go straight
        else:
            # Longest ray is to a side - reward turning that way
            steer_alignment = action_steer * desired_steer_direction  # positive if same direction
            if steer_alignment > 0.1:  # Action steers in correct direction
                reward += 60.0 * min(1.0, steer_alignment)  # Up to 60pts for strong alignment!
            elif steer_alignment < -0.1:  # Action steers AWAY from longest ray
                reward -= 30.0  # Heavy penalty for turning away from open space    # Penalties for hugging walls or obstacles detected by rays
        # BOOST USAGE REWARD/PENALTY
        if action_taken.get('boost'):
            if longest_idx == center_idx and forward_ray > 0.6 and speed > 1.5:
                # Good boost conditions: clear path and some speed already
                reward += 80.0  # Significant bonus for correct boost usage (was 40.0)
                reward += 20.0 * speed  # Scale with speed (was 2.0)
            else:
                # Wasteful / unsafe boost
                reward -= 35.0
    for ray_val in rays:
        if ray_val < 0.15:
            reward -= 8.0
        elif ray_val < 0.3:
            reward -= 3.0
    
    # PHASE 2: Avoid other cars (secondary priority)
    if all_coords:
        car_x, car_y = car.get_position()
        for other_x, other_y in all_coords:
            dx = other_x - car_x
            dy = other_y - car_y
            dist = (dx*dx + dy*dy) ** 0.5
            if dist < 50:  # Close to another car
                reward -= 5.0  # Moderate penalty (was -2)
            elif dist < 100:  # Approaching another car
                reward -= 1.0  # Small warning penalty
    
    # PHASE 3: Progress rewards (tertiary - only after learning safety)
    dprog = max(0.0, progress - prev_progress)
    reward += 50.0 * dprog  # Increased to encourage faster completion
    
    # MICRO-PROGRESS REWARD: Continuous feedback for every pixel traveled (solves reward delay)
    import math
    distance_moved = math.sqrt((car_x - last_pos[0])**2 + (car_y - last_pos[1])**2)
    
    # Only count forward movement (not spinning/reversing)
    if distance_moved > 0 and not collided:
        # Small but continuous reward for any forward movement
        micro_reward = 0.15 * distance_moved  # ~1.5 pts per 10 pixels
        reward += micro_reward
        _CAR_STATE[car_idx]['total_distance'] += distance_moved
    
    # Update position tracking
    _CAR_STATE[car_idx]['last_pos'] = (car_x, car_y)
    
    # Increment checkpoint timer and lap timer
    _CAR_STATE[car_idx]['checkpoint_step_timer'] += 1
    _CAR_STATE[car_idx]['lap_step_timer'] += 1
    
    # Checkpoint milestone reward with multiplicative scaling (immediate feedback)
    # Get actual checkpoint count from game state
    current_checkpoint_count = len(_GAME._checkpoints_collected.get(car_idx, set()))
    last_checkpoint_count = _CAR_STATE[car_idx].get('last_checkpoint_count', 0)
    
    # Detect when a new checkpoint is crossed
    if current_checkpoint_count > last_checkpoint_count:
        _CAR_STATE[car_idx]['checkpoints_passed'] += 1
        _CAR_STATE[car_idx]['last_checkpoint_count'] = current_checkpoint_count
        
        # Get time taken (steps) to reach this checkpoint
        steps_to_checkpoint = _CAR_STATE[car_idx]['checkpoint_step_timer']
        _CAR_STATE[car_idx]['checkpoint_step_timer'] = 0  # Reset timer for next checkpoint
        
        # Apply multiplicative reward: base * (1.5 ^ checkpoint_number) - time_penalty
        # cp1=50, cp2=75, cp3=112.5, cp4=168.75, cp5=253.125, etc.
        base_checkpoint_reward = 100.0  # Increased from 10.0
        multiplier = 1.5 ** (_CAR_STATE[car_idx]['checkpoints_passed'] - 1)
        checkpoint_reward = base_checkpoint_reward * multiplier
        
        # Subtract time penalty (0.05 points per step taken - reduced from 0.1)
        time_penalty = 1.5 * steps_to_checkpoint
        checkpoint_reward = max(-100.0, checkpoint_reward - time_penalty)  
        reward += checkpoint_reward
    
    # LAP COMPLETION BONUS: Reward for finishing a full lap
    current_laps = _GAME._laps_completed.get(car_idx, 0)
    last_laps = _CAR_STATE[car_idx].get('last_lap_count', 0)
    
    if current_laps > last_laps:
        # Lap completed! Give bonus minus time taken
        steps_for_lap = _CAR_STATE[car_idx]['lap_step_timer']
        _CAR_STATE[car_idx]['lap_step_timer'] = 0  # Reset for next lap
        
        base_lap_bonus = 500.0  # Increased from 200.0
        lap_time_penalty = 0.05 * steps_for_lap  # Reduced from 0.1 points per step
        lap_bonus = max(0.0, base_lap_bonus - lap_time_penalty)  # Don't go negative
        
        reward += lap_bonus
        _CAR_STATE[car_idx]['last_lap_count'] = current_laps
    
    # PHASE 4: Speed bonus (lowest priority - only when safe)
    if not collided and dprog > 0:
        # Base speed reward when making progress
        reward += speed  # Increased to encourage faster driving (was 0.15)
        
        # Speed boost when aligned with longest ray (go FAST when path is clear!)
        if longest_idx == center_idx:  # Longest ray is straight ahead
            # Speed bonus when driving straight toward open space
            reward += 8.0 * speed  # Increased to encourage speed when safe (was 5.5)
        elif offset == 1:  # Longest ray is one step away (slight turn needed)
            # Moderate speed bonus
            reward += 4.5 * speed  # Increased (was 2.0)
        else:  # Longest ray is far to the side (sharp turn needed)
            # Small penalty for going too fast when need to turn sharply
            if speed > 3.0:
                reward -= 0.1 * (speed - 2.0)  # Reduced penalty for excess speed (was 0.5)
        
        # Cornering smoothness bonus: reward maintaining speed while turning
        # Higher reward for smooth, fast cornering
        car_obj = _GAME._cars[car_idx]
        steering_magnitude = abs(car_obj._steering_angle) / 100.0  # normalize to [0,1]
        smoothness = 1.0 - steering_magnitude
        reward += 4.0 * speed * smoothness  # Boost smooth cornering reward (was 3.0)
    
    # Bonus for collision-free progress milestones
    if steps_alive > 100 and dprog > 0:
        reward += 5.0  # Bonus for sustained safe driving
    
    # Goal reward (completing lap safely)
    if progress >= 0.99:
        reward += 200.0  # Big reward for finishing (was 100)
        if steps_alive > 500:  # Finished without many crashes
            reward += 100.0  # Extra bonus for clean lap
    
    # Stuck detection: no movement for many steps (position-based, more accurate)
    if distance_moved < 2.0:  # Moved less than 2 pixels = stuck
        _CAR_STATE[car_idx]['stuck_counter'] += 1
        
        # Heavy penalty when car is declared stuck
        if _CAR_STATE[car_idx]['stuck_counter'] >= _STUCK_THRESHOLD:
            reward -= 800.0  # Massive penalty for getting stuck
    else:
        _CAR_STATE[car_idx]['stuck_counter'] = 0
        _CAR_STATE[car_idx]['last_progress'] = progress
    
    # Update tracking
    _CAR_STATE[car_idx]['progress'] = progress
    _CAR_STATE[car_idx]['collided'] = collided
    
    # Track performance metrics for evolutionary selection
    if car_idx not in _CAR_PERFORMANCE:
        _CAR_PERFORMANCE[car_idx] = {'total_distance': 0.0, 'avg_survival': 0.0, 'episodes': 0}
    
    # Track both checkpoint progress and actual distance traveled
    _CAR_PERFORMANCE[car_idx]['total_distance'] += dprog + (distance_moved * 0.001)  # Weight actual distance less than checkpoints
    
    return reward


def train_multi_car(episodes=100, max_steps=3000, gamma=0.99, lr=1e-3, 
                    batch_size=128, target_update_freq=5):  # Optimized hyperparameters
    """Train multiple cars simultaneously using shared Q-network (Code Bullet approach).
    
    All cars contribute to the same replay buffer and share weights,
    which significantly speeds up learning. When all cars are stuck,
    they respawn to continue training.
    """
    import numpy as np
    import tensorflow as tf
    from models.my_model.model import (
        _ensure_q_net, _REPLAY_BUFFER, _ACTIONS, _NUM_ACTIONS, 
        _STATE_DIM, _obs_to_state, save_weights, _Q_NET, _TARGET_Q_NET
    )
    
    # Import and build networks
    from models.my_model import model as model_module
    model_module._ensure_q_net()
    Q_NET = model_module._Q_NET
    TARGET_Q_NET = model_module._TARGET_Q_NET
    EPSILON = model_module._EPSILON
    EPSILON_DECAY = model_module._EPSILON_DECAY
    EPSILON_MIN = model_module._EPSILON_MIN
    
    print(f"\n{'='*60}")
    print(f"MULTI-CAR Q-LEARNING TRAINING (Code Bullet Style)")
    print(f"{'='*60}")
    print(f"Training with {_NUM_CARS} cars simultaneously")
    print(f"Replay buffer size: {_REPLAY_BUFFER.maxlen}")
    print(f"Episodes: {episodes}, Max steps/episode: {max_steps}")
    print(f"Stuck threshold: {_STUCK_THRESHOLD} steps without progress")
    print(f"{'='*60}\n")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    for ep in range(episodes):
        # Initialize environment only once; keep running across episodes
        if ep == 0 or _GAME is None:
            car_states = env_reset_fn_multi()
            states = {i: _obs_to_state(car_states[i]['obs'], car_states[i]['car']) for i in range(_NUM_CARS)}
            car_objects = {i: car_states[i]['car'] for i in range(_NUM_CARS)}
        else:
            # Reuse existing game and cars; refresh observations
            states = {i: _obs_to_state(_GAME._cars[i].get_observation(), _GAME._cars[i]) for i in range(_NUM_CARS)}
            car_objects = {i: _GAME._cars[i] for i in range(_NUM_CARS)}
        ep_rewards = {i: 0.0 for i in range(_NUM_CARS)}
        # Per-episode exploration metrics
        explore_counts = {i: 0 for i in range(_NUM_CARS)}  # times epsilon caused random action
        action_counts = {i: 0 for i in range(_NUM_CARS)}   # total actions selected
        
        for step in range(max_steps):
            # Per-car stuck handling will reset only individual cars now (no global respawn)
            
            # Epsilon-greedy action selection for each car
            actions_by_car = {}
            for car_idx in range(_NUM_CARS):
                action_counts[car_idx] += 1
                if np.random.rand() < EPSILON:
                    # Exploration
                    action_idx = np.random.randint(0, _NUM_ACTIONS)
                    explore_counts[car_idx] += 1
                else:
                    # Exploitation
                    q_vals = Q_NET(tf.expand_dims(states[car_idx], axis=0)).numpy()[0]
                    action_idx = np.argmax(q_vals)
                actions_by_car[car_idx] = _ACTIONS[action_idx]
            
            # Step all cars
            results = env_step_fn_multi(actions_by_car)
            
            # Process each car's result
            for obs, reward, done, car, car_idx in results:
                next_state = _obs_to_state(obs, car)
                ep_rewards[car_idx] += reward
                
                # Store in shared replay buffer (skip non-discrete forced recovery actions)
                action_used = actions_by_car[car_idx]
                try:
                    action_used_idx = _ACTIONS.index(action_used)
                except ValueError:
                    action_used_idx = None
                if action_used_idx is not None:
                    _REPLAY_BUFFER.append((states[car_idx], action_used_idx, reward, next_state, done))
                
                states[car_idx] = next_state
                if done:
                    # Lap finished: do NOT reset car; allow it to continue next lap seamlessly
                    # We still treat this as a terminal for Q update (large reward), but keep position.
                    pass

                # Individual stuck reset
                if _CAR_STATE.get(car_idx, {}).get('stuck_counter', 0) >= _STUCK_THRESHOLD:
                    print(f"  [STUCK] Car{car_idx} resetting (stuck_counter={_CAR_STATE[car_idx]['stuck_counter']})")
                    car = _GAME._cars[car_idx]
                    reset(car)
                    obs = car.get_observation()
                    states[car_idx] = _obs_to_state(obs, car)
                    # Clear progress metrics so it can relearn
                    _CAR_STATE[car_idx]['stuck_counter'] = 0
                    _CAR_STATE[car_idx]['last_progress'] = 0.0
                    _CAR_STATE[car_idx]['progress'] = 0.0
            
            # Train on shared replay buffer (Bellman update)
            if len(_REPLAY_BUFFER) >= batch_size:
                indices = np.random.choice(len(_REPLAY_BUFFER), batch_size, replace=False)
                batch = [_REPLAY_BUFFER[i] for i in indices]
                states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)
                
                states_t = tf.convert_to_tensor(np.stack(states_b), dtype=tf.float32)
                actions_t = tf.convert_to_tensor(actions_b, dtype=tf.int32)
                rewards_t = tf.convert_to_tensor(rewards_b, dtype=tf.float32)
                next_states_t = tf.convert_to_tensor(np.stack(next_states_b), dtype=tf.float32)
                dones_t = tf.convert_to_tensor(dones_b, dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    q_vals = Q_NET(states_t)
                    q_target = tf.identity(q_vals)
                    
                    next_q_vals = TARGET_Q_NET(next_states_t)
                    max_next_q = tf.reduce_max(next_q_vals, axis=1)
                    td_target = rewards_t + gamma * max_next_q * (1.0 - dones_t)
                    
                    for i in range(batch_size):
                        q_target = tf.tensor_scatter_nd_update(
                            q_target,
                            [[i, actions_t[i]]],
                            [td_target[i]]
                        )
                    
                    loss = tf.reduce_mean((q_vals - q_target) ** 2)
                
                grads = tape.gradient(loss, Q_NET.trainable_variables)
                optimizer.apply_gradients(zip(grads, Q_NET.trainable_variables))
        
        # Decay epsilon
        EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
        model_module._EPSILON = EPSILON
        
        # EVOLUTIONARY SELECTION: Keep best performers, eliminate worst
        if _EVOLUTION_ENABLED and (ep + 1) % 5 == 0:  # Every 5 episodes
            # Calculate fitness for each car (distance + survival time)
            fitness_scores = {}
            for car_idx in range(_NUM_CARS):
                perf = _CAR_PERFORMANCE.get(car_idx, {'total_distance': 0.0, 'avg_survival': 0.0, 'episodes': 1})
                survival = _CAR_STATE.get(car_idx, {}).get('steps_alive', 0)
                
                # Update average survival
                perf['episodes'] += 1
                old_avg = perf['avg_survival']
                perf['avg_survival'] = (old_avg * (perf['episodes'] - 1) + survival) / perf['episodes']
                
                # Fitness = distance traveled + survival bonus
                fitness_scores[car_idx] = perf['total_distance'] + perf['avg_survival'] * 0.01
            
            # Sort cars by fitness (best to worst)
            sorted_cars = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
            best_idx = sorted_cars[0][0]
            worst_indices = [idx for idx, score in sorted_cars[-3:]]  # Bottom 3 performers
            
            print(f"  [EVOLUTION] Best: Car{best_idx} (fitness={fitness_scores[best_idx]:.1f})")
            print(f"  [EVOLUTION] Eliminating: {worst_indices} - Resetting to learn from best")
            
            # Reset worst performers' metrics (they start fresh, but network stays shared)
            for idx in worst_indices:
                if idx in _CAR_PERFORMANCE:
                    _CAR_PERFORMANCE[idx]['total_distance'] = 0.0
                    _CAR_PERFORMANCE[idx]['avg_survival'] = 0.0
                    _CAR_PERFORMANCE[idx]['episodes'] = 0
                if idx in _CAR_STATE:
                    _CAR_STATE[idx]['steps_alive'] = 0
                    _CAR_STATE[idx]['progress'] = 0.0
        
        # Update target network
        if (ep + 1) % target_update_freq == 0:
            TARGET_Q_NET.set_weights(Q_NET.get_weights())
        
        # Logging with survival stats and fitness
        avg_reward = sum(ep_rewards.values()) / len(ep_rewards)
        rewards_str = ", ".join([f"Car{i}={ep_rewards[i]:.0f}" for i in range(_NUM_CARS)])
        buffer_size = len(_REPLAY_BUFFER)
        stuck_counts = [_CAR_STATE.get(i, {}).get('stuck_counter', 0) for i in range(_NUM_CARS)]
        survival_times = [_CAR_STATE.get(i, {}).get('steps_alive', 0) for i in range(_NUM_CARS)]
        avg_survival = sum(survival_times) / len(survival_times) if survival_times else 0
        # Exploration percentages per car
        exploration_rates = [ (explore_counts[i] / action_counts[i] * 100.0) if action_counts[i] > 0 else 0.0 for i in range(_NUM_CARS) ]
        exploration_str = ", ".join([f"Car{i}={exploration_rates[i]:.1f}%" for i in range(_NUM_CARS)])
        
        # Show top 3 performers
        if (ep + 1) % 5 == 0 and _EVOLUTION_ENABLED:
            fitness_scores = {}
            for i in range(_NUM_CARS):
                perf = _CAR_PERFORMANCE.get(i, {'total_distance': 0.0, 'avg_survival': 0.0})
                fitness_scores[i] = perf['total_distance'] + perf.get('avg_survival', 0) * 0.01
            top3 = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join([f"Car{idx}({score:.1f})" for idx, score in top3])
            print(f"Ep {ep+1}/{episodes} [{rewards_str}] avg={avg_reward:.1f} survival={avg_survival:.0f} explore=[{exploration_str}] eps={EPSILON:.4f} buf={buffer_size}")
            print(f"  [TOP PERFORMERS] {top_str}")
        else:
            print(f"Ep {ep+1}/{episodes} [{rewards_str}] avg={avg_reward:.1f} survival={avg_survival:.0f} explore=[{exploration_str}] eps={EPSILON:.4f} buf={buffer_size}")
        
        # Calculate performance score: total distance traveled + survival time
        total_distance_covered = sum(
            _CAR_PERFORMANCE.get(i, {}).get('total_distance', 0.0) 
            for i in range(_NUM_CARS)
        )
        total_survival = sum(survival_times)
        performance_score = total_distance_covered * 100 + total_survival  # Weight distance heavily
        
        # Save best model when achieving new best performance (distance + survival)
        global _BEST_PERFORMANCE_SCORE
        if performance_score > _BEST_PERFORMANCE_SCORE:
            old_best = _BEST_PERFORMANCE_SCORE
            _BEST_PERFORMANCE_SCORE = performance_score
            import os
            best_path = os.path.join(os.path.dirname(save_weights.__code__.co_filename), 'best_weights.weights.h5')
            save_weights(path=best_path)
            print(f"  [NEW BEST] distance={total_distance_covered:.2f} survival={total_survival} score={performance_score:.0f} (prev={old_best:.0f})")
        
        # Periodic checkpoint save (every 10 episodes)
        if (ep + 1) % 10 == 0:
            save_weights()
            print(f"  [CHECKPOINT] Weights saved at episode {ep+1}")


if __name__ == "__main__":
    train_multi_car(episodes=100, max_steps=3000, batch_size=128)
