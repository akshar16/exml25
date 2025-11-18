from env.controls import forward, back, steer_right, steer_left, boost, brake
import tensorflow as tf
import numpy as np
import os
from collections import deque

_NO_MOVE_THRESHOLD_STEPS = 120
_REVERSE_RECOVERY_STEPS = 30
_BOT_STATE = {}

def _build_discrete_action_space():
    """12 discrete actions: (throttle, steering, brake) with gradual steering control."""
    return [
        {'throttle': 1.0, 'steer': 0.0, 'brake': False , 'boost': True},          
        {'throttle': 1.0, 'steer': 0.4, 'brake': False, 'boost': True},                                        
        {'throttle': 1.0, 'steer': -0.4, 'brake': False,'boost': True},                                      
        {'throttle': 1.0, 'steer': 1.0, 'brake': False,'boost': True},                                 
        {'throttle': 1.0, 'steer': -1.0, 'brake': False,'boost': True},                               
        {'throttle': 1.0, 'steer': 1.5, 'brake': False},                                      
        {'throttle': 1.0, 'steer': -1.5, 'brake': False},                                    
        {'throttle': 0.0, 'steer': 0.0, 'brake': False},                      
        {'throttle': 0.0, 'steer': 1.0, 'brake': False},                            
        {'throttle': 0.0, 'steer': -1.0, 'brake': False},                          
        {'throttle': 0.0, 'steer': 0.0, 'brake': True},                         
        {'throttle': 1.0, 'steer': 0.0, 'brake': True, 'boost': True},                                   
        {'throttle': 1.0, 'steer': 0.0, 'brake': False, 'boost': True},                              
    ]


_RAYCAST_DISTANCE_STEPS = np.linspace(0.0, 300.0, 25).astype(np.float32)
_ANGLE_CACHE = {}

def _raycast_sensors(car, num_rays=5, max_distance=300.0):
    """Measure distances to track walls using directional ray sensors (optimized).

    Optimizations:
    - Precomputed distance steps (_RAYCAST_DISTANCE_STEPS)
    - Angle sin/cos caching per (car_angle, offset)
    - Early break when collision detected
    """
    import math

    x, y = car.get_position()
    base_angle = car._angle
    track = car._track
    game = getattr(car, "_game", None)

    if num_rays == 5:
        angles = (-90.0, -45.0, 0.0, 45.0, 90.0)
    elif num_rays == 9:
        angles = (-135.0, -90.0, -45.0, -27.5, 0.0, 27.5, 45.0, 90.0, 135.0)
    elif num_rays == 12:
        angles = (-150.0, -120.0, -90.0, -60.0, -30.0, -15.0, 0.0, 15.0, 30.0, 60.0, 90.0, 120.0)
    else:
        angles = tuple(np.linspace(-90.0, 90.0, num_rays, dtype=np.float32))

    distances_out = np.empty(len(angles), dtype=np.float32)

    for idx, offset in enumerate(angles):
        key = (int(base_angle), offset)                                           
        if key not in _ANGLE_CACHE:
            rad = math.radians(base_angle + offset)
            _ANGLE_CACHE[key] = (math.sin(rad), math.cos(rad))
        s, c = _ANGLE_CACHE[key]

        hit_distance = None
        for dist in _RAYCAST_DISTANCE_STEPS:
                                                                      
            check_x = float(x + dist * s)
            check_y = float(y - dist * c)
                                        
            blocked = False
            if track.check_collision(check_x, check_y):
                blocked = True
            elif hasattr(track, "is_drivable"):
                try:
                    drivable = track.is_drivable(check_x, check_y)
                except TypeError:
                    drivable = True
                if not drivable:
                    blocked = True
            if blocked:
                hit_distance = dist
                break
            if game is not None:
                blocked = False
                for other in game._cars:
                    if other is car:
                        continue
                                                                         
                    try:
                        if other._hitbox.collidepoint(int(check_x), int(check_y)):
                            blocked = True
                            break
                    except Exception:
                                                                    
                        continue
                if blocked:
                    hit_distance = dist
                    break
        distances_out[idx] = 1.0 if hit_distance is None else (hit_distance / max_distance)

    return distances_out


def _obs_to_state(obs, car):
    """Encode observation into state vector using ray sensors and car detection.

    Uses 12-ray expanded sensor layout for richer spatial awareness.
    """
    import math
    
                                                                                       
    rays = _raycast_sensors(car, num_rays=12, max_distance=300.0)
    
                                         
    speed = float(obs.get('speed', 0.0)) / 5.0                          
    lap_progress = float(obs.get('lap_progress', 0.0))
    
                                                            
    all_coords = obs.get('all_coords', [])
    car_x, car_y = car.get_position()
    car_angle = car._angle
    
    if all_coords:
                          
        min_dist = float('inf')
        nearest_angle = 0.0
        
        for other_x, other_y in all_coords:
            dx = other_x - car_x
            dy = other_y - car_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < min_dist:
                min_dist = dist
                                                       
                angle_to_car = math.degrees(math.atan2(dx, -dy))                             
                relative_angle = angle_to_car - car_angle
                                          
                while relative_angle > 180:
                    relative_angle -= 360
                while relative_angle < -180:
                    relative_angle += 360
                nearest_angle = relative_angle / 180.0                        
        
        nearest_car_dist = min(1.0, min_dist / 300.0)                                    
    else:
        nearest_car_dist = 1.0                          
        nearest_angle = 0.0
    
    state = np.concatenate([rays, [speed, lap_progress, nearest_car_dist, nearest_angle]], axis=0)
    return state.astype(np.float32)


def _build_q_network(state_dim, num_actions):
    """Q-network: state -> Q-values for each action.
    
    Deeper architecture with residual-style skip connections for better learning:
    - 4 hidden layers with increasing then decreasing capacity
    - Dropout for regularization
    - More parameters = better racing strategy learning
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(state_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),                                        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_actions, activation=None)
    ])
    return model


                                                 
_Q_NET = None
_TARGET_Q_NET = None
_REPLAY_BUFFER = deque(maxlen=100000)
_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'q_weights.weights.h5')
_EPSILON = 0.275                              
_EPSILON_DECAY = 0.9970                                        
_EPSILON_MIN = 0.01
_ACTIONS = _build_discrete_action_space()
_NUM_ACTIONS = len(_ACTIONS)                                    
_STATE_DIM = 16                                                                         


def _ensure_q_net():
    global _Q_NET, _TARGET_Q_NET
    if _Q_NET is None:
        print("[INIT] Building Q-network...")
        _Q_NET = _build_q_network(_STATE_DIM, _NUM_ACTIONS)
        _Q_NET(tf.zeros((1, _STATE_DIM)))                           
        _TARGET_Q_NET = _build_q_network(_STATE_DIM, _NUM_ACTIONS)
        _TARGET_Q_NET(tf.zeros((1, _STATE_DIM)))
        _TARGET_Q_NET.set_weights(_Q_NET.get_weights())
        print(f"[INIT] Q-network built. State dim: {_STATE_DIM}, Actions: {_NUM_ACTIONS}")
        print(f"[INIT] Weights will be saved to: {_WEIGHTS_PATH}")
        if os.path.exists(_WEIGHTS_PATH):
            try:
                _Q_NET.load_weights(_WEIGHTS_PATH)
                _TARGET_Q_NET.set_weights(_Q_NET.get_weights())
                print(f"[INIT] Loaded existing weights from: {_WEIGHTS_PATH}")
            except Exception as e:
                print(f"[INIT] Failed to load weights: {e}")


def save_weights(path=None):
    """Save Q-network weights."""
    path = path or _WEIGHTS_PATH
    _ensure_q_net()
    _Q_NET.save_weights(path)
    print(f"[SAVE] Q-network weights saved to: {path}")


def load_weights(path=None):
    """Load Q-network weights."""
    path = path or _WEIGHTS_PATH
    _ensure_q_net()
    if os.path.exists(path):
        _Q_NET.load_weights(path)
        print(f"[LOAD] Q-network weights loaded from: {path}")
        return True
    else:
        print(f"[LOAD] No weights found at: {path}, starting fresh")
        return False


def model(car):
    """Inference entry: greedy policy (no exploration during inference)."""
    try:
        obs = car.get_observation()
    except Exception:
        return

    cid = id(car)
    x, y = car.get_position()
    st = _BOT_STATE.setdefault(cid, {
        'last_pos': (x, y),
        'no_move_steps': 0,
        'reverse_timer': 0,
    })

    last_x, last_y = st['last_pos']
    dx = x - last_x
    dy = y - last_y
    moved = (dx*dx + dy*dy) ** 0.5
    if moved < 1.0:
        st['no_move_steps'] += 1
    else:
        st['no_move_steps'] = 0
    st['last_pos'] = (x, y)

    if st['reverse_timer'] > 0 or st['no_move_steps'] >= _NO_MOVE_THRESHOLD_STEPS:
        if st['reverse_timer'] <= 0:
            st['reverse_timer'] = _REVERSE_RECOVERY_STEPS
            st['no_move_steps'] = 0
        st['reverse_timer'] -= 1
        try:
            back(car)
        except Exception:
            pass
        return
    
    state = _obs_to_state(obs, car)
    _ensure_q_net()
    
                                                               
    q_vals = _Q_NET(tf.expand_dims(state, axis=0)).numpy()[0]
    action_idx = np.argmax(q_vals)
    action = _ACTIONS[action_idx]
    
                                           
    if action['throttle'] > 0.5:
        forward(car)
    elif action['throttle'] < -0.5:
        back(car)
    
    if action['steer'] > 0.5:
        steer_right(car)
    elif action['steer'] < -0.5:
        steer_left(car)
    
    if action['brake']:
        brake(car)
                                             
    if action.get('boost'):
        try:
            boost(car)
        except Exception:
            pass


def train_q_learning(env_reset_fn, env_step_fn, episodes=500, max_steps=2000,
                      gamma=0.99, lr=1e-3, batch_size=32, target_update_freq=5):
    """Train Q-network using experience replay and Bellman updates.
    
    This implements the Q-learning algorithm as described in Code Bullet's
    "AI Learns to Drive" video:
    - Epsilon-greedy exploration: initially random, gradually becoming greedy.
    - Experience replay: store transitions and train on mini-batches.
    - Bellman equation: Q(s,a) = R + gamma * max Q(s',a').
    - Target network: use a separate network for stability.
    
    Parameters:
    - env_reset_fn: callable() -> (obs_dict, car_ref)
    - env_step_fn: callable(action_dict) -> (next_obs_dict, reward, done, car_ref)
    - episodes: number of training episodes
    - max_steps: max steps per episode
    - gamma: discount factor (0.99 = look ahead ~100 steps)
    - lr: learning rate for Adam optimizer
    - batch_size: replay buffer batch size
    - target_update_freq: update target network every N episodes
    """
    global _EPSILON
    
    _ensure_q_net()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    for ep in range(episodes):
        obs, car = env_reset_fn()
        state = _obs_to_state(obs, car)
        ep_reward = 0.0
        
        for step in range(max_steps):
                                                                          
            if np.random.rand() < _EPSILON:
                action_idx = np.random.randint(0, _NUM_ACTIONS)
            else:
                q_vals = _Q_NET(tf.expand_dims(state, axis=0)).numpy()[0]
                action_idx = np.argmax(q_vals)
            
            action = _ACTIONS[action_idx]
            next_obs, reward, done, car = env_step_fn(action)
            next_state = _obs_to_state(next_obs, car)
            ep_reward += reward
            
                                               
            _REPLAY_BUFFER.append((state, action_idx, reward, next_state, done))
            
                                                                     
            if len(_REPLAY_BUFFER) >= batch_size:
                                   
                indices = np.random.choice(len(_REPLAY_BUFFER), batch_size, replace=False)
                batch = [_REPLAY_BUFFER[i] for i in indices]
                states, actions, rewards, next_states, dones = zip(*batch)
                
                           
                states = tf.convert_to_tensor(np.stack(states), dtype=tf.float32)
                actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                next_states = tf.convert_to_tensor(np.stack(next_states), dtype=tf.float32)
                dones = tf.convert_to_tensor(dones, dtype=tf.float32)
                
                                                                       
                with tf.GradientTape() as tape:
                    q_vals = _Q_NET(states)
                    q_target = tf.identity(q_vals)
                    
                                                      
                    next_q_vals = _TARGET_Q_NET(next_states)
                    max_next_q = tf.reduce_max(next_q_vals, axis=1)
                    td_target = rewards + gamma * max_next_q * (1.0 - dones)
                    
                                                          
                    for i in range(batch_size):
                        q_target = tf.tensor_scatter_nd_update(
                            q_target,
                            [[i, actions[i]]],
                            [td_target[i]]
                        )
                    
                              
                    loss = tf.reduce_mean((q_vals - q_target) ** 2)
                
                                 
                grads = tape.gradient(loss, _Q_NET.trainable_variables)
                optimizer.apply_gradients(zip(grads, _Q_NET.trainable_variables))
            
            state = next_state
            if done:
                break
        
                                                         
        _EPSILON = max(_EPSILON_MIN, _EPSILON * _EPSILON_DECAY)
        
                                                              
        if (ep + 1) % target_update_freq == 0:
            _TARGET_Q_NET.set_weights(_Q_NET.get_weights())
        
                                   
       
        # Decay epsilon (exploration decreases over time)
        _EPSILON = max(_EPSILON_MIN, _EPSILON * _EPSILON_DECAY)
        
        # Update target network periodically (stability trick)
        if (ep + 1) % target_update_freq == 0:
            _TARGET_Q_NET.set_weights(_Q_NET.get_weights())
        
        # Logging and checkpointing
        print(f"Episode {ep+1}/{episodes} steps={step+1} reward={ep_reward:.1f} epsilon={_EPSILON:.4f}")
        if (ep + 1) % 10 == 0:
            save_weights()


