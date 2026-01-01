# Learn2Drive - Self-Driving Car RL Simulation

A 2D self-driving car simulation that learns to navigate procedurally generated racetracks using reinforcement learning. Implements and compares three RL algorithms: **PPO**, **DQN**, and **GRPO**.

## Features

- **Pygame-based 2D simulation** with top-down view
- **Procedural track generation**: Random tracks via Catmull-Rom splines, or classic oval
- **Realistic car physics**: acceleration, steering, friction, high-speed handling
- **Lidar-like sensors**: 9 rays detecting distance to track boundaries
- **Three RL algorithms** for comparison:
  - **PPO** (Proximal Policy Optimization) - continuous actions via stable-baselines3
  - **DQN** (Deep Q-Network) - discrete actions via stable-baselines3
  - **GRPO** (Group Relative Policy Optimization) - custom PyTorch implementation
- **O(1) collision detection** via precomputed bitmap
- **Stuck detection**: Episodes terminate if car stops moving
- **Visual debugging**: waypoints, target direction, sensor rays
- **Manual demo mode** with keyboard control

## Installation

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pygame gymnasium stable-baselines3 numpy torch
```

## Usage

### Demo Mode (Manual Control)
Test the environment with keyboard controls:
```bash
python self_driving_car.py demo                    # Random track
python self_driving_car.py demo --track oval       # Oval track
python self_driving_car.py demo --track-seed 42    # Reproducible random track
```
Controls: `UP/W`=Accelerate, `DOWN/S`=Brake, `LEFT/A`=Left, `RIGHT/D`=Right, `R`=Reset, `N`=New track, `ESC`=Quit

### Training
```bash
# Train with PPO (recommended)
python self_driving_car.py train --algo ppo --timesteps 300000

# Train with DQN
python self_driving_car.py train --algo dqn --timesteps 300000

# Train with GRPO
python self_driving_car.py train --algo grpo

# Train on specific track
python self_driving_car.py train --algo ppo --track-seed 42

# Train on oval track
python self_driving_car.py train --algo ppo --track oval

# Quick training (50k steps) for testing
python self_driving_car.py quick --algo ppo

# Train with visualization (slower)
python self_driving_car.py train --algo ppo --render
```

### Evaluation
```bash
# Evaluate a trained model
python self_driving_car.py evaluate --algo ppo

# Evaluate on specific track
python self_driving_car.py evaluate --algo ppo --track-seed 123

# Evaluate all algorithms
python self_driving_car.py evaluate --algo all
```

### Algorithm Comparison
Train and compare all three algorithms:
```bash
python self_driving_car.py compare --timesteps 100000
```

## Command Line Options

```
python self_driving_car.py [mode] [options]

Modes:
  train     - Train an agent
  evaluate  - Evaluate a trained agent
  demo      - Manual keyboard control
  compare   - Compare all three algorithms
  quick     - Quick training (50k steps)

Options:
  --algo {ppo,dqn,grpo,all}    Algorithm to use (default: ppo)
  --timesteps N                 Total training timesteps (default: 200000)
  --episodes N                  Number of evaluation episodes (default: 5)
  --render                      Render during training
  --track {random,oval}         Track type (default: random)
  --track-seed N                Seed for reproducible tracks
  --randomize-tracks            New random track each episode
```

## Algorithm Comparison

| Algorithm | Action Space | Key Characteristics |
|-----------|-------------|---------------------|
| **PPO** | Continuous | Stable, sample-efficient, smooth control |
| **DQN** | Discrete (8 actions) | Value-based, replay buffer, all-forward actions |
| **GRPO** | Continuous | No critic needed, group-relative advantages |

## Environment Details

### State Space (11 dimensions)
- Normalized speed [-1, 1]
- Angle difference to next waypoint [-1, 1]
- 9 lidar ray distances [0, 1]

### Action Space
- **Continuous** (PPO/GRPO): [throttle, steering] each in [-1, 1]
- **Discrete** (DQN): 8 predefined throttle/steering combinations (all forward-biased)

### Physics Parameters
- Max speed: 12.0 units/frame
- Acceleration: 0.4 units/frame
- Turn rate: 10 degrees/frame at full steering
- Friction: 0.008 (low for maintaining speed)

### Reward Function (Speed-Focused)
- Base speed reward: Linear with speed ratio
- Quadratic speed bonus: Rewards high speed exponentially
- High speed bonus: +0.4 when above 70% max speed
- Max speed bonus: +0.6 when above 90% max speed
- Waypoint progress: +5.0 per waypoint
- Lap completion: +500
- Collision penalty: -30
- Stopped penalty: -0.5

## Project Structure

```
learn2drive/
├── self_driving_car.py    # Main simulation and training code
├── README.md              # This file
├── CLAUDE.md              # AI assistant instructions
├── VIBE_HISTORY.md        # Development history
├── ppo_car_model.zip      # Trained PPO model (after training)
├── dqn_car_model.zip      # Trained DQN model (after training)
└── grpo_car_model.pt      # Trained GRPO model (after training)
```

## Hyperparameters

Key hyperparameters in the `HYPERPARAMS` dictionary:

| Parameter | Value | Description |
|-----------|-------|-------------|
| total_timesteps | 200,000 | Default training duration |
| learning_rate | 3e-4 | Learning rate for all algorithms |
| max_speed | 12.0 | Maximum car speed |
| acceleration | 0.4 | Acceleration rate |
| turn_rate | 10.0 | Degrees per frame at full steering |
| num_lidar_rays | 9 | Number of sensor rays |
| reward_lap_complete | 500 | Bonus for completing a lap |

## Performance

- Training throughput: ~450-500 steps/sec (PPO), ~1200-2600 steps/sec (DQN)
- Collision detection: O(1) via precomputed bitmap
- Trained models reach 100% max speed on oval track

## License

MIT License
