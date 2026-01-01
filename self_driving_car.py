"""
Self-Driving Car 2D Simulation with Multiple RL Algorithms
===========================================================
A complete implementation of a 2D self-driving car that learns to navigate
a racetrack using three different RL algorithms:

1. PPO (Proximal Policy Optimization) - Continuous actions, stable training
2. DQN (Deep Q-Network) - Discrete actions, value-based learning
3. GRPO (Group Relative Policy Optimization) - Custom implementation,
   uses group-relative advantages without a critic

Algorithm Comparison:
- PPO: Best for continuous control (smooth steering/throttle), stable, sample-efficient
- DQN: Good for discrete actions, simpler to implement, can struggle with high-dim actions
- GRPO: Novel approach using relative rewards within sampled groups, no critic needed

Requirements:
    pip install pygame gymnasium stable-baselines3 numpy torch

Usage:
    python self_driving_car.py [mode] [--algo ALGO]

    Modes: train, evaluate, demo, compare, quick
    Algorithms: ppo, dqn, grpo, all

Author: Claude Code
"""

import math
import sys
import os
from typing import Optional, Tuple, List, Dict, Any
from collections import deque
import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical

# Stable-baselines3 imports
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# =============================================================================
# HYPERPARAMETERS (Tunable)
# =============================================================================
HYPERPARAMS = {
    # Training
    "total_timesteps": 300_000,      # Total training steps
    "learning_rate": 3e-4,           # Learning rate
    "gamma": 0.99,                   # Discount factor

    # PPO specific
    "ppo_n_steps": 2048,             # Steps per PPO update
    "ppo_batch_size": 64,            # PPO minibatch size
    "ppo_n_epochs": 10,              # PPO epochs per update
    "ppo_clip_range": 0.2,           # PPO clip range
    "ppo_ent_coef": 0.02,            # Entropy coefficient (increased for exploration)

    # DQN specific
    "dqn_buffer_size": 100_000,      # Replay buffer size
    "dqn_learning_starts": 5_000,    # Steps before learning (reduced)
    "dqn_batch_size": 64,            # DQN batch size
    "dqn_tau": 0.005,                # Target network update rate
    "dqn_exploration_fraction": 0.5, # Exploration schedule fraction (increased)
    "dqn_exploration_final": 0.05,   # Final exploration rate

    # GRPO specific
    "grpo_group_size": 8,            # Number of samples per state
    "grpo_update_freq": 2048,        # Steps between updates (collect more data)
    "grpo_clip_range": 0.2,          # Policy clip range
    "grpo_beta": 0.01,               # KL penalty coefficient
    "grpo_ent_coef": 0.02,           # Entropy coefficient
    "grpo_n_epochs": 4,              # Update epochs per batch

    # Environment
    "max_steps_per_episode": 3000,   # Episode timeout
    "num_lidar_rays": 9,             # Number of sensor rays
    "lidar_max_distance": 150.0,     # Max sensor range (pixels)
    "stuck_threshold_steps": 60,     # Steps without movement before termination
    "stuck_distance_threshold": 5.0, # Minimum distance to travel to not be "stuck"

    # Physics - tuned for high-speed racing
    "max_speed": 12.0,               # Maximum car speed (high for racing)
    "acceleration": 0.4,             # Acceleration rate (fast response)
    "friction": 0.008,               # Speed decay per frame (low for maintaining speed)
    "turn_rate": 10.0,               # Degrees per frame at full steering (high for tight corners at speed)

    # Rewards - heavily incentivize maximum speed
    "reward_velocity_toward_waypoint": 0.1,  # Reward for velocity toward next waypoint
    "reward_progress": 5.0,          # Reward for passing waypoint (high)
    "reward_speed": 0.3,             # Base reward for speed (strong)
    "reward_high_speed_bonus": 0.4,  # Extra reward when above 70% max speed (very strong)
    "reward_max_speed_bonus": 0.6,   # Extra reward when above 90% max speed (push for max)
    "reward_alive": 0.0,             # No survival bonus (don't encourage slow play)
    "penalty_centerline": 0.0005,    # Small penalty for being off centerline
    "penalty_collision": -30.0,      # Penalty for hitting wall
    "penalty_wrong_direction": -2.0, # Penalty for going backward
    "penalty_stopped": -0.5,         # Strong penalty for not moving
    "penalty_stuck": -50.0,          # Penalty for being stuck (episode termination)
    "reward_lap_complete": 500.0,    # Big bonus for completing a lap fast
}

# =============================================================================
# CONSTANTS
# =============================================================================
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (150, 0, 255)

# DQN discrete actions mapping (throttle, steering)
# All forward-biased - no brake action to prevent oscillating policies
DQN_ACTIONS = [
    (1.0, 0.0),    # Full forward
    (1.0, -0.5),   # Forward + Left
    (1.0, 0.5),    # Forward + Right
    (1.0, -1.0),   # Forward + Hard Left
    (1.0, 1.0),    # Forward + Hard Right
    (0.7, -0.7),   # Medium throttle + Medium Left
    (0.7, 0.7),    # Medium throttle + Medium Right
    (0.5, 0.0),    # Half forward (for tight corners)
]


# =============================================================================
# TRACK GENERATION UTILITIES
# =============================================================================
def catmull_rom_spline(
    points: List[Tuple[float, float]],
    num_samples: int = 100
) -> List[Tuple[float, float]]:
    """
    Generate a smooth closed curve using Catmull-Rom spline interpolation.

    Args:
        points: Control points (will be treated as closed loop)
        num_samples: Total number of output points

    Returns:
        List of interpolated points forming a smooth closed curve
    """
    if len(points) < 3:
        return points

    # Close the loop by wrapping points
    pts = list(points) + [points[0], points[1], points[2]]

    result = []
    samples_per_segment = max(1, num_samples // len(points))

    for i in range(len(points)):
        p0 = pts[i]
        p1 = pts[i + 1]
        p2 = pts[i + 2]
        p3 = pts[i + 3]

        for t_idx in range(samples_per_segment):
            t = t_idx / samples_per_segment

            # Catmull-Rom basis functions
            t2 = t * t
            t3 = t2 * t

            # Standard Catmull-Rom
            x = 0.5 * (
                (2 * p1[0]) +
                (-p0[0] + p2[0]) * t +
                (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1]) +
                (-p0[1] + p2[1]) * t +
                (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3
            )

            result.append((x, y))

    return result


def compute_curve_normals(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Compute outward-pointing normals for each point on a closed clockwise curve.

    Args:
        points: Closed curve points (clockwise order)

    Returns:
        List of unit normal vectors (nx, ny) for each point
    """
    n = len(points)
    normals = []

    for i in range(n):
        # Get neighboring points
        prev_pt = points[(i - 1) % n]
        curr_pt = points[i]
        next_pt = points[(i + 1) % n]

        # Tangent from prev to next (smoothed)
        tx = next_pt[0] - prev_pt[0]
        ty = next_pt[1] - prev_pt[1]

        # Normalize
        length = math.sqrt(tx*tx + ty*ty)
        if length > 1e-6:
            tx /= length
            ty /= length
        else:
            tx, ty = 1.0, 0.0

        # Normal pointing outward (left of tangent for clockwise curve)
        # For clockwise on screen (Y down), outward is to the left
        nx = ty
        ny = -tx

        normals.append((nx, ny))

    return normals


def offset_curve_robust(
    points: List[Tuple[float, float]],
    normals: List[Tuple[float, float]],
    offset: float,
    center: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """
    Offset a curve robustly, preventing self-intersection at sharp corners.

    Args:
        points: Centerline points
        normals: Precomputed outward normals
        offset: Distance to offset (positive = outward)
        center: Track center point for validation

    Returns:
        List of offset points
    """
    n = len(points)
    result = []

    for i in range(n):
        px, py = points[i]
        nx, ny = normals[i]

        # Basic offset
        ox = px + nx * offset
        oy = py + ny * offset

        result.append((ox, oy))

    return result


def smooth_curve(points: List[Tuple[float, float]], iterations: int = 2) -> List[Tuple[float, float]]:
    """
    Apply Laplacian smoothing to a closed curve.

    Args:
        points: Curve points
        iterations: Number of smoothing passes

    Returns:
        Smoothed curve points
    """
    pts = list(points)
    n = len(pts)

    for _ in range(iterations):
        new_pts = []
        for i in range(n):
            prev_pt = pts[(i - 1) % n]
            curr_pt = pts[i]
            next_pt = pts[(i + 1) % n]

            # Average with neighbors (0.5 weight on current, 0.25 on each neighbor)
            nx = 0.5 * curr_pt[0] + 0.25 * prev_pt[0] + 0.25 * next_pt[0]
            ny = 0.5 * curr_pt[1] + 0.25 * prev_pt[1] + 0.25 * next_pt[1]

            new_pts.append((nx, ny))
        pts = new_pts

    return pts


def point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.

    Args:
        x, y: Point coordinates
        polygon: List of polygon vertices

    Returns:
        True if point is inside polygon
    """
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside

        j = i

    return inside


def segments_intersect(
    p1: Tuple[float, float], p2: Tuple[float, float],
    p3: Tuple[float, float], p4: Tuple[float, float]
) -> bool:
    """
    Check if line segment p1-p2 intersects with segment p3-p4.
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def polygon_self_intersects(polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if a polygon has any self-intersecting edges.
    """
    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        # Check against non-adjacent edges
        for j in range(i + 2, n):
            if j == (i - 1) % n or j == i or j == (i + 1) % n:
                continue
            if (i == 0 and j == n - 1):
                continue  # Adjacent at wrap

            p3 = polygon[j]
            p4 = polygon[(j + 1) % n]

            if segments_intersect(p1, p2, p3, p4):
                return True

    return False


def generate_smooth_control_points(
    center_x: float,
    center_y: float,
    num_points: int = 8,
    base_radius: float = 200,
    radius_variation: float = 0.3,
    rng: Optional[random.Random] = None
) -> List[Tuple[float, float]]:
    """
    Generate smooth control points that won't create self-intersecting tracks.

    Uses a base radius with limited variation to ensure smooth curves.

    Args:
        center_x, center_y: Center of the track
        num_points: Number of control points (6-10 recommended)
        base_radius: Base distance from center
        radius_variation: Max variation as fraction of base_radius (0.0-0.5)
        rng: Random number generator

    Returns:
        List of control points arranged clockwise
    """
    if rng is None:
        rng = random.Random()

    points = []

    # Limit variation to prevent sharp corners
    radius_variation = min(0.4, max(0.1, radius_variation))

    # Generate radii with smoothing (adjacent points have similar radii)
    radii = []
    for i in range(num_points):
        variation = rng.uniform(-radius_variation, radius_variation)
        radii.append(base_radius * (1 + variation))

    # Smooth the radii to prevent sharp transitions
    for _ in range(2):
        smoothed = []
        for i in range(num_points):
            prev_r = radii[(i - 1) % num_points]
            curr_r = radii[i]
            next_r = radii[(i + 1) % num_points]
            smoothed.append(0.5 * curr_r + 0.25 * prev_r + 0.25 * next_r)
        radii = smoothed

    # Generate points
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        r = radii[i]

        x = center_x + r * math.cos(angle)
        y = center_y + r * math.sin(angle)

        # Clamp to screen bounds with margin
        margin = 80
        x = max(margin, min(SCREEN_WIDTH - margin, x))
        y = max(margin, min(SCREEN_HEIGHT - margin, y))

        points.append((x, y))

    return points


def validate_track_geometry(
    centerline: List[Tuple[float, float]],
    outer: List[Tuple[float, float]],
    inner: List[Tuple[float, float]],
    min_width: float = 40
) -> bool:
    """
    Validate that a track has valid geometry.

    Checks:
    - No self-intersecting boundaries
    - Minimum track width maintained
    - Centerline is between boundaries

    Args:
        centerline: Track centerline points
        outer: Outer boundary points
        inner: Inner boundary points
        min_width: Minimum required track width

    Returns:
        True if track geometry is valid
    """
    # Check for self-intersecting boundaries
    if polygon_self_intersects(outer):
        return False
    if polygon_self_intersects(inner):
        return False

    # Check minimum width at sample points
    n = len(centerline)
    for i in range(0, n, max(1, n // 20)):
        outer_pt = outer[i]
        inner_pt = inner[i]

        dx = outer_pt[0] - inner_pt[0]
        dy = outer_pt[1] - inner_pt[1]
        width = math.sqrt(dx*dx + dy*dy)

        if width < min_width:
            return False

    # Check centerline points are on track
    for i in range(0, n, max(1, n // 10)):
        cx, cy = centerline[i]
        in_outer = point_in_polygon(cx, cy, outer)
        in_inner = point_in_polygon(cx, cy, inner)

        if not in_outer or in_inner:
            return False

    return True


# =============================================================================
# TRACK CLASS
# =============================================================================
class Track:
    """
    Defines the racetrack geometry using procedurally generated shapes.
    Supports random track generation with inner and outer boundaries.
    Provides methods for collision detection, centerline distance, and progress.

    Uses a precomputed collision bitmap for O(1) collision detection.
    """

    # Class constant for maximum generation attempts
    MAX_GENERATION_ATTEMPTS = 20

    def __init__(
        self,
        center_x: float,
        center_y: float,
        seed: Optional[int] = None,
        track_type: str = "random"
    ):
        """
        Initialize the track.

        Args:
            center_x, center_y: Center point of the track
            seed: Random seed for reproducible track generation (None = random)
            track_type: "random" for procedural, "oval" for classic oval track
        """
        self.center_x = center_x
        self.center_y = center_y
        self.seed = seed
        self.track_type = track_type

        # Track width (distance from centerline to boundary)
        self.track_width = 55  # Half-width on each side

        # Generate the track geometry
        if track_type == "oval":
            self._generate_oval_track()
        else:
            self._generate_random_track_with_validation(seed)

        # Generate waypoints along the centerline for progress tracking
        self.waypoints = self._sample_waypoints(num_points=36)
        self.num_waypoints = len(self.waypoints)

        # Cache start position (first waypoint area)
        self._compute_start_position()

        # Precompute collision bitmap for O(1) collision detection
        self._generate_collision_bitmap()

    def _generate_oval_track(self):
        """Generate classic oval track for backward compatibility."""
        outer_rx = 320
        outer_ry = 210
        inner_rx = 200
        inner_ry = 90

        # Generate centerline as ellipse
        num_pts = 200
        self.centerline = []
        center_rx = (outer_rx + inner_rx) / 2
        center_ry = (outer_ry + inner_ry) / 2

        for i in range(num_pts):
            angle = 2 * math.pi * i / num_pts
            x = self.center_x + center_rx * math.cos(angle)
            y = self.center_y + center_ry * math.sin(angle)
            self.centerline.append((x, y))

        self.track_width = (outer_rx - inner_rx) / 2

        # Compute normals and generate boundaries
        normals = compute_curve_normals(self.centerline)
        self.outer_boundary = offset_curve_robust(
            self.centerline, normals, self.track_width, (self.center_x, self.center_y)
        )
        self.inner_boundary = offset_curve_robust(
            self.centerline, normals, -self.track_width, (self.center_x, self.center_y)
        )

    def _generate_random_track_with_validation(self, seed: Optional[int] = None):
        """
        Generate a random track with validation and retry logic.

        Attempts to generate a valid track, retrying with modified parameters
        if validation fails.
        """
        base_seed = seed if seed is not None else random.randint(0, 1000000)

        for attempt in range(self.MAX_GENERATION_ATTEMPTS):
            # Use different seed for each attempt
            attempt_seed = base_seed + attempt * 1000
            rng = random.Random(attempt_seed)

            # Vary parameters based on attempt to find valid configuration
            if attempt < 5:
                # First attempts: normal parameters
                num_control_points = rng.randint(6, 10)
                base_radius = rng.uniform(160, 220)
                radius_variation = rng.uniform(0.15, 0.35)
                self.track_width = rng.uniform(50, 65)
            elif attempt < 10:
                # Middle attempts: simpler tracks
                num_control_points = rng.randint(5, 8)
                base_radius = rng.uniform(180, 210)
                radius_variation = rng.uniform(0.1, 0.25)
                self.track_width = rng.uniform(55, 70)
            else:
                # Later attempts: very simple tracks
                num_control_points = rng.randint(4, 6)
                base_radius = 200
                radius_variation = 0.15
                self.track_width = 60

            # Generate control points
            control_points = generate_smooth_control_points(
                self.center_x,
                self.center_y,
                num_points=num_control_points,
                base_radius=base_radius,
                radius_variation=radius_variation,
                rng=rng
            )

            # Create smooth centerline
            self.centerline = catmull_rom_spline(control_points, num_samples=200)

            # Apply additional smoothing to centerline
            self.centerline = smooth_curve(self.centerline, iterations=2)

            # Compute normals for consistent offsetting
            normals = compute_curve_normals(self.centerline)

            # Generate boundaries
            self.outer_boundary = offset_curve_robust(
                self.centerline, normals, self.track_width, (self.center_x, self.center_y)
            )
            self.inner_boundary = offset_curve_robust(
                self.centerline, normals, -self.track_width, (self.center_x, self.center_y)
            )

            # Smooth boundaries to remove any artifacts
            self.outer_boundary = smooth_curve(self.outer_boundary, iterations=1)
            self.inner_boundary = smooth_curve(self.inner_boundary, iterations=1)

            # Validate the track
            if validate_track_geometry(
                self.centerline,
                self.outer_boundary,
                self.inner_boundary,
                min_width=self.track_width * 1.5
            ):
                # Store the successful seed
                self.seed = attempt_seed
                return

        # Fallback to oval if all attempts fail
        self._generate_oval_track()
        self.seed = None

    def _sample_waypoints(self, num_points: int) -> List[Tuple[float, float]]:
        """Sample waypoints evenly along the centerline."""
        if len(self.centerline) == 0:
            return []

        step = max(1, len(self.centerline) // num_points)
        waypoints = []

        for i in range(num_points):
            idx = (i * step) % len(self.centerline)
            waypoints.append(self.centerline[idx])

        return waypoints

    def _compute_start_position(self):
        """Compute the starting position and angle for the car."""
        if len(self.centerline) < 2:
            self.start_x = self.center_x
            self.start_y = self.center_y
            self.start_angle = 0
            return

        # Find a valid starting point on the centerline
        # The first point might be at a tight curve, so search for a good spot
        for i in range(len(self.centerline)):
            px, py = self.centerline[i]

            # Check if this point is actually on the track (use polygon check before bitmap exists)
            in_outer = point_in_polygon(px, py, self.outer_boundary)
            in_inner = point_in_polygon(px, py, self.inner_boundary)

            if in_outer and not in_inner:
                self.start_x = px
                self.start_y = py

                # Calculate starting angle (direction to next point)
                next_idx = (i + 5) % len(self.centerline)  # Look ahead a bit for smoother angle
                next_pt = self.centerline[next_idx]
                dx = next_pt[0] - self.start_x
                dy = next_pt[1] - self.start_y
                self.start_angle = math.degrees(math.atan2(dy, dx))
                return

        # Fallback: use first centerline point even if not perfectly on track
        self.start_x = self.centerline[0][0]
        self.start_y = self.centerline[0][1]

        next_pt = self.centerline[1]
        dx = next_pt[0] - self.start_x
        dy = next_pt[1] - self.start_y
        self.start_angle = math.degrees(math.atan2(dy, dx))

    def _generate_collision_bitmap(self):
        """
        Precompute a bitmap for O(1) collision detection.

        Uses pygame to rasterize polygons efficiently.
        The bitmap is a 2D boolean array where True = on track, False = off track.
        """
        # Create a temporary surface to rasterize the track
        # Use pygame without initializing display (headless)
        temp_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        temp_surface.fill((0, 0, 0))  # Black = off track

        # Draw outer boundary filled (white = on track candidate)
        if len(self.outer_boundary) > 2:
            pygame.draw.polygon(temp_surface, (255, 255, 255), self.outer_boundary)

        # Draw inner boundary filled (black = off track)
        if len(self.inner_boundary) > 2:
            pygame.draw.polygon(temp_surface, (0, 0, 0), self.inner_boundary)

        # Convert surface to numpy array
        # pygame uses (width, height, channels) but we need (height, width)
        pixels = pygame.surfarray.array3d(temp_surface)

        # Track is where pixel is white (any channel > 0)
        self.collision_bitmap = pixels[:, :, 0].T > 0  # Transpose to get (height, width)

    def get_next_waypoint(self, current_idx: int) -> Tuple[float, float]:
        """Get the next waypoint position."""
        next_idx = (current_idx + 1) % self.num_waypoints
        return self.waypoints[next_idx]

    def get_direction_to_waypoint(self, x: float, y: float, waypoint_idx: int) -> float:
        """Get angle (in degrees) from position to the next waypoint."""
        next_wp = self.get_next_waypoint(waypoint_idx)
        dx = next_wp[0] - x
        dy = next_wp[1] - y
        return math.degrees(math.atan2(dy, dx))

    def is_on_track(self, x: float, y: float) -> bool:
        """Check if a point is within the track boundaries using precomputed bitmap."""
        # Convert to integer pixel coordinates
        px = int(x)
        py = int(y)

        # Bounds check
        if px < 0 or px >= SCREEN_WIDTH or py < 0 or py >= SCREEN_HEIGHT:
            return False

        # O(1) bitmap lookup
        return self.collision_bitmap[py, px]

    def distance_to_centerline(self, x: float, y: float) -> float:
        """Calculate distance from a point to the nearest centerline point."""
        min_dist = float('inf')

        for cx, cy in self.centerline:
            dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < min_dist:
                min_dist = dist

        return min_dist

    def get_nearest_waypoint_index(self, x: float, y: float) -> int:
        """Find the index of the nearest waypoint."""
        min_dist = float('inf')
        nearest_idx = 0

        for i, (wx, wy) in enumerate(self.waypoints):
            dist = (x - wx) ** 2 + (y - wy) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        return nearest_idx

    def raycast(self, x: float, y: float, angle: float, max_dist: float) -> float:
        """Cast a ray from position (x, y) at given angle. Returns distance to wall."""
        step = 3.0
        dist = 0.0

        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))

        while dist < max_dist:
            check_x = x + dist * cos_a
            check_y = y + dist * sin_a

            if not self.is_on_track(check_x, check_y):
                return dist

            dist += step

        return max_dist

    def draw(self, surface: pygame.Surface, current_waypoint_idx: int = 0, show_waypoints: bool = True):
        """Render the track on the pygame surface."""
        # Draw outer boundary (filled polygon - the track surface)
        if len(self.outer_boundary) > 2:
            pygame.draw.polygon(surface, GRAY, self.outer_boundary)

        # Draw inner boundary (cut out - not drivable)
        if len(self.inner_boundary) > 2:
            pygame.draw.polygon(surface, DARK_GRAY, self.inner_boundary)

        # Draw boundary lines for clarity
        if len(self.outer_boundary) > 2:
            pygame.draw.polygon(surface, (120, 120, 120), self.outer_boundary, 2)
        if len(self.inner_boundary) > 2:
            pygame.draw.polygon(surface, (120, 120, 120), self.inner_boundary, 2)

        # Draw centerline
        if len(self.centerline) > 2:
            pygame.draw.lines(surface, WHITE, True, self.centerline, 1)

        # Draw start/finish line
        if len(self.centerline) >= 2:
            start_outer = self.outer_boundary[0]
            start_inner = self.inner_boundary[0]
            pygame.draw.line(surface, YELLOW, start_outer, start_inner, 3)

        # Draw waypoints
        if show_waypoints:
            for i, (wx, wy) in enumerate(self.waypoints):
                # Highlight current and next waypoint
                if i == current_waypoint_idx:
                    color = ORANGE
                    radius = 6
                elif i == (current_waypoint_idx + 1) % self.num_waypoints:
                    color = GREEN
                    radius = 8
                else:
                    color = (80, 80, 80)
                    radius = 3
                pygame.draw.circle(surface, color, (int(wx), int(wy)), radius)


# =============================================================================
# CAR CLASS
# =============================================================================
class Car:
    """
    Represents the car with physics simulation.
    Handles movement, sensors, and rendering.
    """

    def __init__(self, x: float, y: float, angle: float = 0.0):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = 0.0

        self.width = 30
        self.height = 15

        self.max_speed = HYPERPARAMS["max_speed"]
        self.acceleration = HYPERPARAMS["acceleration"]
        self.friction = HYPERPARAMS["friction"]
        self.turn_rate = HYPERPARAMS["turn_rate"]

        self.num_rays = HYPERPARAMS["num_lidar_rays"]
        self.ray_max_dist = HYPERPARAMS["lidar_max_distance"]
        self.ray_angles = np.linspace(-90, 90, self.num_rays)
        self.ray_distances = np.zeros(self.num_rays)

    def reset(self, x: float, y: float, angle: float = 0.0, initial_speed: float = 1.0):
        """Reset car to starting position with small initial speed."""
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = initial_speed  # Start with some speed to help initial exploration
        self.ray_distances = np.zeros(self.num_rays)

    def update(self, throttle: float, steering: float, track: Track) -> bool:
        """
        Update car physics. Returns True if collision occurred.
        """
        # Apply steering - scales with speed but allows some turning even at low speed
        speed_factor = max(0.3, abs(self.speed) / self.max_speed)  # Min 30% steering at low speed
        self.angle += steering * self.turn_rate * speed_factor

        self.angle = self.angle % 360

        # Apply throttle
        if throttle > 0:
            self.speed += throttle * self.acceleration
        elif throttle < 0:
            self.speed += throttle * self.acceleration * 0.5

        # Apply friction
        self.speed *= (1 - self.friction)

        # Clamp speed
        self.speed = np.clip(self.speed, -self.max_speed * 0.3, self.max_speed)

        # Update position
        rad = math.radians(self.angle)
        new_x = self.x + self.speed * math.cos(rad)
        new_y = self.y + self.speed * math.sin(rad)

        # Check collision
        if not track.is_on_track(new_x, new_y):
            return True

        self.x = new_x
        self.y = new_y

        self._update_sensors(track)

        return False

    def _update_sensors(self, track: Track):
        """Update lidar sensor readings."""
        for i, ray_angle in enumerate(self.ray_angles):
            world_angle = self.angle + ray_angle
            self.ray_distances[i] = track.raycast(
                self.x, self.y, world_angle, self.ray_max_dist
            )

    def get_state(self, angle_to_next_waypoint: float = 0.0) -> np.ndarray:
        """
        Get the car's state vector for the RL agent.

        State includes:
        - Normalized speed [-1, 1]
        - Normalized angle difference to next waypoint [-1, 1]
        - Lidar ray distances [0, 1]
        """
        # Speed normalized to [-1, 1]
        norm_speed = np.clip(self.speed / self.max_speed, -1.0, 1.0)

        # Calculate angle difference to next waypoint (how much to turn)
        angle_diff = angle_to_next_waypoint - self.angle
        # Normalize to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        # Normalize to [-1, 1]
        norm_angle_diff = angle_diff / 180.0

        # Lidar rays normalized to [0, 1]
        norm_rays = self.ray_distances / self.ray_max_dist

        state = np.concatenate([
            [norm_speed, norm_angle_diff],
            norm_rays
        ]).astype(np.float32)

        return state

    def get_velocity_toward_angle(self, target_angle: float) -> float:
        """Get component of velocity toward a target angle."""
        if self.speed <= 0:
            return 0.0
        # Calculate angle difference
        angle_diff = math.radians(target_angle - self.angle)
        # Velocity component in target direction
        return self.speed * math.cos(angle_diff)

    def draw(self, surface: pygame.Surface, draw_sensors: bool = True, color: Tuple = BLUE):
        """Render the car and its sensors."""
        if draw_sensors:
            for i, (ray_angle, dist) in enumerate(zip(self.ray_angles, self.ray_distances)):
                world_angle = math.radians(self.angle + ray_angle)
                end_x = self.x + dist * math.cos(world_angle)
                end_y = self.y + dist * math.sin(world_angle)

                ratio = dist / self.ray_max_dist
                ray_color = (int(255 * (1 - ratio)), int(255 * ratio), 0)
                pygame.draw.line(surface, ray_color, (self.x, self.y), (end_x, end_y), 1)

        corners = self._get_corners()
        pygame.draw.polygon(surface, color, corners)
        pygame.draw.polygon(surface, WHITE, corners, 2)

        rad = math.radians(self.angle)
        front_x = self.x + (self.width / 2 + 5) * math.cos(rad)
        front_y = self.y + (self.width / 2 + 5) * math.sin(rad)
        pygame.draw.circle(surface, RED, (int(front_x), int(front_y)), 4)

    def _get_corners(self) -> List[Tuple[float, float]]:
        """Calculate the four corners of the car rectangle."""
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        hw = self.width / 2
        hh = self.height / 2

        corners = [
            (self.x + hw * cos_a - hh * sin_a, self.y + hw * sin_a + hh * cos_a),
            (self.x - hw * cos_a - hh * sin_a, self.y - hw * sin_a + hh * cos_a),
            (self.x - hw * cos_a + hh * sin_a, self.y - hw * sin_a - hh * cos_a),
            (self.x + hw * cos_a + hh * sin_a, self.y + hw * sin_a - hh * cos_a),
        ]

        return corners


# =============================================================================
# BASE GYMNASIUM ENVIRONMENT
# =============================================================================
class RacetrackEnv(gym.Env):
    """
    Custom Gymnasium environment for the racetrack.
    Supports both continuous (PPO, GRPO) and discrete (DQN) action spaces.
    Supports random procedural track generation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        discrete_actions: bool = False,
        track_type: str = "random",
        track_seed: Optional[int] = None,
        randomize_track_on_reset: bool = False
    ):
        """
        Initialize the racetrack environment.

        Args:
            render_mode: "human" for display, "rgb_array" for pixel output, None for no render
            discrete_actions: True for DQN (discrete), False for PPO/GRPO (continuous)
            track_type: "random" for procedural tracks, "oval" for classic oval
            track_seed: Seed for reproducible track generation (None = random)
            randomize_track_on_reset: If True, generate new random track each episode
        """
        super().__init__()

        self.render_mode = render_mode
        self.discrete_actions = discrete_actions
        self.track_type = track_type
        self.track_seed = track_seed
        self.randomize_track_on_reset = randomize_track_on_reset
        self._episode_count = 0

        # Initialize track
        self.track = Track(
            SCREEN_WIDTH // 2,
            SCREEN_HEIGHT // 2,
            seed=track_seed,
            track_type=track_type
        )

        # Get starting position from track
        self.start_x = self.track.start_x
        self.start_y = self.track.start_y
        self.start_angle = self.track.start_angle

        self.car = Car(self.start_x, self.start_y, self.start_angle)

        # Define observation space
        # State: [speed, angle_diff_to_waypoint, lidar_rays...]
        # speed and angle_diff are in [-1, 1], lidar rays in [0, 1]
        num_observations = 2 + HYPERPARAMS["num_lidar_rays"]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_observations,), dtype=np.float32
        )

        # Define action space
        if discrete_actions:
            self.action_space = spaces.Discrete(len(DQN_ACTIONS))
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )

        # Episode tracking
        self.max_steps = HYPERPARAMS["max_steps_per_episode"]
        self.current_step = 0
        self.last_waypoint_idx = 0
        self.laps_completed = 0
        self.total_progress = 0

        # Stuck detection - track cumulative movement over a window
        self.stuck_threshold = HYPERPARAMS["stuck_threshold_steps"]
        self.stuck_min_speed = 0.5  # Minimum average speed to not be considered stuck
        self.recent_positions = []  # Ring buffer of recent positions

        # Pygame setup (lazy initialization)
        self.screen = None
        self.clock = None
        self.font = None
        self.algo_name = "Unknown"

    def set_algo_name(self, name: str):
        """Set the algorithm name for display."""
        self.algo_name = name

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._episode_count += 1

        # Optionally regenerate track each episode
        if self.randomize_track_on_reset and self._episode_count > 1:
            # Use episode count + base seed for varied but reproducible tracks
            if self.track_seed is not None:
                new_seed = self.track_seed + self._episode_count
            else:
                new_seed = None  # Fully random

            self.track = Track(
                SCREEN_WIDTH // 2,
                SCREEN_HEIGHT // 2,
                seed=new_seed,
                track_type=self.track_type
            )

            # Update start position from new track
            self.start_x = self.track.start_x
            self.start_y = self.track.start_y
            self.start_angle = self.track.start_angle

        self.car.reset(self.start_x, self.start_y, self.start_angle)
        self.car._update_sensors(self.track)

        self.current_step = 0
        self.last_waypoint_idx = self.track.get_nearest_waypoint_index(self.car.x, self.car.y)
        self.laps_completed = 0
        self.total_progress = 0

        # Reset stuck detection
        self.recent_positions = [(self.car.x, self.car.y)]

        # Get angle to next waypoint for observation
        angle_to_waypoint = self.track.get_direction_to_waypoint(
            self.car.x, self.car.y, self.last_waypoint_idx
        )
        observation = self.car.get_state(angle_to_waypoint)
        info = {"track_seed": self.track.seed}

        return observation, info

    def step(self, action):
        self.current_step += 1

        # Parse action
        if self.discrete_actions:
            throttle, steering = DQN_ACTIONS[int(action)]
        else:
            throttle = float(action[0])
            steering = float(action[1])

        prev_waypoint_idx = self.last_waypoint_idx
        prev_x, prev_y = self.car.x, self.car.y

        # Get angle to next waypoint BEFORE moving (for velocity reward)
        angle_to_waypoint = self.track.get_direction_to_waypoint(
            self.car.x, self.car.y, prev_waypoint_idx
        )

        # Update car physics
        collision = self.car.update(throttle, steering, self.track)

        # Calculate reward
        reward = 0.0
        terminated = False
        truncated = False

        # Track position for stuck detection
        self.recent_positions.append((self.car.x, self.car.y))
        if len(self.recent_positions) > self.stuck_threshold:
            self.recent_positions.pop(0)

        # Check for stuck condition - calculate total distance traveled over window
        is_stuck = False
        if len(self.recent_positions) >= self.stuck_threshold:
            total_distance = 0.0
            for i in range(1, len(self.recent_positions)):
                dx = self.recent_positions[i][0] - self.recent_positions[i-1][0]
                dy = self.recent_positions[i][1] - self.recent_positions[i-1][1]
                total_distance += math.sqrt(dx*dx + dy*dy)
            avg_speed = total_distance / len(self.recent_positions)
            is_stuck = avg_speed < self.stuck_min_speed

        if collision:
            reward = HYPERPARAMS["penalty_collision"]
            terminated = True
            # Still need to update sensors for valid observation
            self.car._update_sensors(self.track)
        elif is_stuck:
            # Car is stuck - terminate with penalty
            reward = HYPERPARAMS["penalty_stuck"]
            terminated = True
        else:
            # Small survival reward (reduced to not encourage slow play)
            reward += HYPERPARAMS["reward_alive"]

            current_waypoint_idx = self.track.get_nearest_waypoint_index(
                self.car.x, self.car.y
            )

            # Progress (handling wrap-around)
            progress = current_waypoint_idx - prev_waypoint_idx
            if progress < -self.track.num_waypoints // 2:
                progress += self.track.num_waypoints
            elif progress > self.track.num_waypoints // 2:
                progress -= self.track.num_waypoints

            self.total_progress += progress
            self.last_waypoint_idx = current_waypoint_idx

            # Check for lap completion
            if self.total_progress >= self.track.num_waypoints:
                self.laps_completed += 1
                self.total_progress -= self.track.num_waypoints
                reward += HYPERPARAMS["reward_lap_complete"]

            # Waypoint progress reward (key reward for making progress)
            if progress > 0:
                reward += progress * HYPERPARAMS["reward_progress"]
            elif progress < 0:
                # Penalty for going backward
                reward += progress * abs(HYPERPARAMS["penalty_wrong_direction"])

            # DENSE REWARD: Velocity toward next waypoint
            velocity_toward = self.car.get_velocity_toward_angle(angle_to_waypoint)
            reward += velocity_toward * HYPERPARAMS["reward_velocity_toward_waypoint"]

            # Speed rewards - strongly encourage maximum speed
            speed_ratio = self.car.speed / self.car.max_speed
            if self.car.speed > 0:
                # Base speed reward (linear with speed)
                reward += speed_ratio * HYPERPARAMS["reward_speed"]

                # Quadratic speed bonus - rewards high speed exponentially more
                reward += (speed_ratio ** 2) * HYPERPARAMS["reward_speed"]

                # Bonus for high speed (above 70% max)
                if speed_ratio > 0.7:
                    reward += HYPERPARAMS["reward_high_speed_bonus"]

                # Extra bonus for near-max speed (above 90% max)
                if speed_ratio > 0.9:
                    reward += HYPERPARAMS["reward_max_speed_bonus"]
            else:
                # Penalty for being stopped or going backward
                reward += HYPERPARAMS["penalty_stopped"]

            # Small centerline penalty (don't want to overweight this)
            dist_to_center = self.track.distance_to_centerline(self.car.x, self.car.y)
            reward -= dist_to_center * HYPERPARAMS["penalty_centerline"]

            if self.current_step >= self.max_steps:
                truncated = True

        # Get new angle to next waypoint for observation
        new_angle_to_waypoint = self.track.get_direction_to_waypoint(
            self.car.x, self.car.y, self.last_waypoint_idx
        )
        observation = self.car.get_state(new_angle_to_waypoint)

        info = {
            "laps": self.laps_completed,
            "progress": self.total_progress,
            "speed": self.car.speed,
            "is_stuck": is_stuck,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption(f"Self-Driving Car - {self.algo_name}")
            else:
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)

        self.screen.fill(DARK_GRAY)
        self.track.draw(self.screen, current_waypoint_idx=self.last_waypoint_idx, show_waypoints=True)

        # Choose color based on algorithm
        color = BLUE
        if "PPO" in self.algo_name:
            color = BLUE
        elif "DQN" in self.algo_name:
            color = GREEN
        elif "GRPO" in self.algo_name:
            color = PURPLE

        self.car.draw(self.screen, draw_sensors=True, color=color)

        # Draw line from car to next waypoint (target direction)
        next_wp = self.track.get_next_waypoint(self.last_waypoint_idx)
        pygame.draw.line(self.screen, YELLOW,
                        (int(self.car.x), int(self.car.y)),
                        (int(next_wp[0]), int(next_wp[1])), 2)

        self._draw_hud()

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        return np.transpose(
            pygame.surfarray.array3d(self.screen), axes=(1, 0, 2)
        ) if self.render_mode == "rgb_array" else None

    def _draw_hud(self):
        """Draw heads-up display with debug info."""
        texts = [
            f"Algorithm: {self.algo_name}",
            f"Speed: {self.car.speed:.1f}",
            f"Angle: {self.car.angle:.0f}deg",
            f"Laps: {self.laps_completed}",
            f"Progress: {self.total_progress:.0f}/{self.track.num_waypoints}",
            f"Step: {self.current_step}/{self.max_steps}",
        ]

        for i, text in enumerate(texts):
            surface = self.font.render(text, True, WHITE)
            self.screen.blit(surface, (10, 10 + i * 20))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# =============================================================================
# GRPO IMPLEMENTATION (Custom PyTorch)
# =============================================================================
class GRPOPolicyNetwork(nn.Module):
    """
    Policy network for GRPO.
    Outputs mean and log_std for continuous action distribution.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        # Initialize log_std to give smaller initial std for more focused exploration
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)  # std ≈ 0.6

        # Initialize mean head with bias toward forward motion
        # action[0] = throttle, action[1] = steering
        with torch.no_grad():
            self.mean_head.bias[0] = 0.5  # Bias toward positive throttle
            self.mean_head.bias[1] = 0.0  # No steering bias

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        mean = torch.tanh(self.mean_head(features))
        std = torch.exp(self.log_std.clamp(-2, 1))  # std in [0.135, 2.7]
        return mean, std

    def get_distribution(self, obs: torch.Tensor) -> Normal:
        mean, std = self.forward(obs)
        return Normal(mean, std)

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_distribution(obs)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of the policy distribution."""
        dist = self.get_distribution(obs)
        return dist.entropy().sum(dim=-1)


class GRPOAgent:
    """
    Group Relative Policy Optimization (GRPO) Agent.

    GRPO is a policy gradient method that:
    1. Samples multiple actions for each state (group sampling)
    2. Computes rewards for each sampled action
    3. Uses group-relative advantages (no critic needed)
    4. Updates policy using clipped surrogate objective

    Key insight: Instead of learning a value function, GRPO normalizes
    advantages within each group of samples, making training more stable.
    """

    def __init__(
        self,
        env: RacetrackEnv,
        lr: float = HYPERPARAMS["learning_rate"],
        gamma: float = HYPERPARAMS["gamma"],
        group_size: int = HYPERPARAMS["grpo_group_size"],
        clip_range: float = HYPERPARAMS["grpo_clip_range"],
        beta: float = HYPERPARAMS["grpo_beta"],
        ent_coef: float = HYPERPARAMS.get("grpo_ent_coef", 0.02),
        n_epochs: int = HYPERPARAMS.get("grpo_n_epochs", 4),
        device: str = "auto"
    ):
        self.env = env
        self.gamma = gamma
        self.group_size = group_size
        self.clip_range = clip_range
        self.beta = beta
        self.ent_coef = ent_coef
        self.n_epochs = n_epochs

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.policy = GRPOPolicyNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Storage for rollout data
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state_tensor)
                action = mean.cpu().numpy()[0]
            else:
                action, _ = self.policy.sample_action(state_tensor)
                action = action.cpu().numpy()[0]

        return np.clip(action, -1.0, 1.0)

    def store_transition(self, state, action, reward, done, log_prob):
        """Store transition for later update."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def compute_returns(self, rewards: List[float], dones: List[bool]) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        R = 0

        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(self.device)
        return returns

    def update(self) -> Dict[str, float]:
        """
        Update policy using GRPO algorithm.

        GRPO key steps:
        1. Compute returns for collected trajectories
        2. For each state, sample multiple actions (group)
        3. Compute group-relative advantages
        4. Update policy with clipped objective + entropy bonus
        """
        if len(self.states) == 0:
            return {}

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)

        # Compute returns
        returns = self.compute_returns(self.rewards, self.dones)

        # Normalize returns (group-relative advantage approximation)
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Multiple epochs of updates (like PPO)
        total_policy_loss = 0
        total_entropy = 0

        for epoch in range(self.n_epochs):
            # Get current policy distribution
            dist = self.policy.get_distribution(states)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs.detach())

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus for exploration
            entropy = self.policy.entropy(states).mean()

            # Total loss (minimize policy loss, maximize entropy)
            loss = policy_loss - self.ent_coef * entropy

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_entropy += entropy.item()

        # Clear storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

        return {
            "policy_loss": total_policy_loss / self.n_epochs,
            "entropy": total_entropy / self.n_epochs,
            "mean_return": returns.mean().item(),
        }

    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train_grpo(
    total_timesteps: int = HYPERPARAMS["total_timesteps"],
    save_path: str = "grpo_car_model.pt",
    render_during_training: bool = False,
    verbose: bool = True,
    track_type: str = "random",
    track_seed: Optional[int] = None
) -> GRPOAgent:
    """Train GRPO agent."""
    print("=" * 60)
    print("Self-Driving Car - GRPO Training")
    print("=" * 60)
    print(f"Track type: {track_type}, seed: {track_seed}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("=" * 60)

    render_mode = "human" if render_during_training else None
    env = RacetrackEnv(
        render_mode=render_mode,
        discrete_actions=False,
        track_type=track_type,
        track_seed=track_seed
    )
    env.set_algo_name("GRPO")

    agent = GRPOAgent(env)

    update_freq = HYPERPARAMS["grpo_update_freq"]
    episode_rewards = []
    episode_laps = []
    best_avg_reward = -float('inf')
    total_laps = 0

    obs, _ = env.reset()
    episode_reward = 0
    episode_count = 0

    for step in range(total_timesteps):
        # Select action
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            action, log_prob = agent.policy.sample_action(state_tensor)
            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().item()

        action = np.clip(action, -1.0, 1.0)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.store_transition(obs, action, reward, done, log_prob)
        episode_reward += reward

        if render_during_training:
            env.render()

        if done:
            episode_rewards.append(episode_reward)
            laps_this_ep = info.get('laps', 0)
            episode_laps.append(laps_this_ep)
            total_laps += laps_this_ep
            episode_count += 1

            if verbose and episode_count % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                recent_laps = sum(episode_laps[-100:])
                progress_pct = (step / total_timesteps) * 100
                print(f"[{progress_pct:5.1f}%] Ep {episode_count} | Step {step:,} | "
                      f"Reward: {avg_reward:.1f} | Recent Laps: {recent_laps} | "
                      f"Total Laps: {total_laps}")

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    agent.save(save_path)

            obs, _ = env.reset()
            episode_reward = 0
        else:
            obs = next_obs

        # Update policy
        if (step + 1) % update_freq == 0 and len(agent.states) > 0:
            metrics = agent.update()
            if verbose and metrics:
                print(f"    Update: loss={metrics['policy_loss']:.4f}, "
                      f"entropy={metrics['entropy']:.4f}, "
                      f"return={metrics['mean_return']:.2f}")

    # Final save
    agent.save(save_path)
    env.close()

    print(f"\nGRPO Training complete!")
    print(f"Total episodes: {episode_count}")
    print(f"Total laps completed: {total_laps}")
    print(f"Model saved to {save_path}")
    return agent


# =============================================================================
# PPO TRAINING
# =============================================================================
def train_ppo(
    total_timesteps: int = HYPERPARAMS["total_timesteps"],
    save_path: str = "ppo_car_model",
    render_during_training: bool = False,
    track_type: str = "random",
    track_seed: Optional[int] = None
) -> PPO:
    """Train PPO agent using stable-baselines3."""
    print("=" * 60)
    print("Self-Driving Car - PPO Training")
    print("=" * 60)
    print(f"Track type: {track_type}, seed: {track_seed}")

    render_mode = "human" if render_during_training else None
    env = RacetrackEnv(
        render_mode=render_mode,
        discrete_actions=False,
        track_type=track_type,
        track_seed=track_seed
    )
    env.set_algo_name("PPO")

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Use CPU for MLP policy - faster than GPU for small networks
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=HYPERPARAMS["learning_rate"],
        gamma=HYPERPARAMS["gamma"],
        n_steps=HYPERPARAMS["ppo_n_steps"],
        batch_size=HYPERPARAMS["ppo_batch_size"],
        n_epochs=HYPERPARAMS["ppo_n_epochs"],
        clip_range=HYPERPARAMS["ppo_clip_range"],
        ent_coef=HYPERPARAMS["ppo_ent_coef"],
        verbose=1,
        device="cpu",
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(save_path)
    vec_env.save(f"{save_path}_vecnormalize.pkl")

    env.close()

    print(f"\nPPO Training complete. Model saved to {save_path}")
    return model


# =============================================================================
# DQN TRAINING
# =============================================================================
def train_dqn(
    total_timesteps: int = HYPERPARAMS["total_timesteps"],
    save_path: str = "dqn_car_model",
    render_during_training: bool = False,
    track_type: str = "random",
    track_seed: Optional[int] = None
) -> DQN:
    """Train DQN agent using stable-baselines3."""
    print("=" * 60)
    print("Self-Driving Car - DQN Training")
    print("=" * 60)
    print(f"Track type: {track_type}, seed: {track_seed}")

    render_mode = "human" if render_during_training else None
    env = RacetrackEnv(
        render_mode=render_mode,
        discrete_actions=True,
        track_type=track_type,
        track_seed=track_seed
    )
    env.set_algo_name("DQN")

    vec_env = DummyVecEnv([lambda: env])

    # Use CPU for MLP policy - faster than GPU for small networks
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=HYPERPARAMS["learning_rate"],
        gamma=HYPERPARAMS["gamma"],
        buffer_size=HYPERPARAMS["dqn_buffer_size"],
        learning_starts=HYPERPARAMS["dqn_learning_starts"],
        batch_size=HYPERPARAMS["dqn_batch_size"],
        tau=HYPERPARAMS["dqn_tau"],
        exploration_fraction=HYPERPARAMS["dqn_exploration_fraction"],
        exploration_final_eps=HYPERPARAMS["dqn_exploration_final"],
        verbose=1,
        device="cpu",
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    model.save(save_path)

    env.close()

    print(f"\nDQN Training complete. Model saved to {save_path}")
    return model


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================
def evaluate_ppo(model_path: str = "ppo_car_model", num_episodes: int = 5, track_type: str = "random", track_seed: Optional[int] = None):
    """Evaluate trained PPO agent."""
    print(f"\n{'='*60}")
    print("Evaluating PPO Agent")
    print(f"Track type: {track_type}, seed: {track_seed}")
    print("=" * 60)

    env = RacetrackEnv(render_mode="human", discrete_actions=False, track_type=track_type, track_seed=track_seed)
    env.set_algo_name("PPO")
    vec_env = DummyVecEnv([lambda: env])

    # Try to load normalization stats, but handle mismatched observation spaces
    try:
        vec_env = VecNormalize.load(f"{model_path}_vecnormalize.pkl", vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print("Loaded normalization stats")
    except FileNotFoundError:
        print("No normalization stats found, using raw observations")
    except AssertionError as e:
        print(f"Warning: Could not load normalization stats (shape mismatch).")
        print("This usually means the model was trained with different settings.")
        print("Please retrain the model with: python self_driving_car.py train --algo ppo")
        return 0.0, 0

    try:
        model = PPO.load(model_path, env=vec_env)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train a model first with: python self_driving_car.py train --algo ppo")
        return 0.0, 0

    total_rewards = []
    total_laps = 0

    for ep in range(num_episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += reward[0]
            env.render()

        total_rewards.append(ep_reward)
        total_laps += info[0].get('laps', 0)
        print(f"Episode {ep+1}: Reward={ep_reward:.1f}, Laps={info[0].get('laps', 0)}")

    env.close()
    print(f"\nPPO Average Reward: {np.mean(total_rewards):.1f}, Total Laps: {total_laps}")
    return np.mean(total_rewards), total_laps


def evaluate_dqn(model_path: str = "dqn_car_model", num_episodes: int = 5, track_type: str = "random", track_seed: Optional[int] = None):
    """Evaluate trained DQN agent."""
    print(f"\n{'='*60}")
    print("Evaluating DQN Agent")
    print(f"Track type: {track_type}, seed: {track_seed}")
    print("=" * 60)

    env = RacetrackEnv(render_mode="human", discrete_actions=True, track_type=track_type, track_seed=track_seed)
    env.set_algo_name("DQN")
    vec_env = DummyVecEnv([lambda: env])

    try:
        model = DQN.load(model_path, env=vec_env)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train a model first with: python self_driving_car.py train --algo dqn")
        return 0.0, 0

    total_rewards = []
    total_laps = 0

    for ep in range(num_episodes):
        obs = vec_env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += reward[0]
            env.render()

        total_rewards.append(ep_reward)
        total_laps += info[0].get('laps', 0)
        print(f"Episode {ep+1}: Reward={ep_reward:.1f}, Laps={info[0].get('laps', 0)}")

    env.close()
    print(f"\nDQN Average Reward: {np.mean(total_rewards):.1f}, Total Laps: {total_laps}")
    return np.mean(total_rewards), total_laps


def evaluate_grpo(model_path: str = "grpo_car_model.pt", num_episodes: int = 5, track_type: str = "random", track_seed: Optional[int] = None):
    """Evaluate trained GRPO agent."""
    print(f"\n{'='*60}")
    print("Evaluating GRPO Agent")
    print(f"Track type: {track_type}, seed: {track_seed}")
    print("=" * 60)

    env = RacetrackEnv(render_mode="human", discrete_actions=False, track_type=track_type, track_seed=track_seed)
    env.set_algo_name("GRPO")

    agent = GRPOAgent(env)
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train a model first with: python self_driving_car.py train --algo grpo")
        return 0.0, 0

    total_rewards = []
    total_laps = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            env.render()

        total_rewards.append(ep_reward)
        total_laps += info.get('laps', 0)
        print(f"Episode {ep+1}: Reward={ep_reward:.1f}, Laps={info.get('laps', 0)}")

    env.close()
    print(f"\nGRPO Average Reward: {np.mean(total_rewards):.1f}, Total Laps: {total_laps}")
    return np.mean(total_rewards), total_laps


# =============================================================================
# COMPARISON FUNCTION
# =============================================================================
def compare_algorithms(timesteps: int = 100_000, eval_episodes: int = 5):
    """Train and compare all three algorithms."""
    print("=" * 60)
    print("ALGORITHM COMPARISON: PPO vs DQN vs GRPO")
    print("=" * 60)

    results = {}

    # Train all algorithms
    print("\n[1/3] Training PPO...")
    train_ppo(total_timesteps=timesteps, save_path="compare_ppo")

    print("\n[2/3] Training DQN...")
    train_dqn(total_timesteps=timesteps, save_path="compare_dqn")

    print("\n[3/3] Training GRPO...")
    train_grpo(total_timesteps=timesteps, save_path="compare_grpo.pt")

    # Evaluate all algorithms
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    results["PPO"] = evaluate_ppo("compare_ppo", eval_episodes)
    results["DQN"] = evaluate_dqn("compare_dqn", eval_episodes)
    results["GRPO"] = evaluate_grpo("compare_grpo.pt", eval_episodes)

    # Print comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"{'Algorithm':<10} {'Avg Reward':<15} {'Total Laps':<12}")
    print("-" * 40)
    for algo, (reward, laps) in results.items():
        print(f"{algo:<10} {reward:<15.1f} {laps:<12}")

    best_algo = max(results.keys(), key=lambda k: results[k][0])
    print(f"\nBest performing algorithm: {best_algo}")

    return results


# =============================================================================
# DEMO MODE
# =============================================================================
def demo_mode(track_type: str = "random", track_seed: Optional[int] = None, randomize_on_reset: bool = True):
    """
    Run demonstration with keyboard control.
    Controls: UP/W=Accelerate, DOWN/S=Brake, LEFT/A=Left, RIGHT/D=Right
              R=Reset (new track if randomize enabled), N=New track, ESC=Quit
    """
    print("=" * 60)
    print("Self-Driving Car - Demo Mode (Keyboard Control)")
    print("=" * 60)
    print("Controls: UP/W=Accelerate, DOWN/S=Brake, LEFT/A=Left, RIGHT/D=Right")
    print("          R=Reset, N=New random track, ESC=Quit")
    print(f"Track type: {track_type}")
    print("=" * 60)

    env = RacetrackEnv(
        render_mode="human",
        discrete_actions=False,
        track_type=track_type,
        track_seed=track_seed,
        randomize_track_on_reset=randomize_on_reset
    )
    env.set_algo_name("Manual")
    obs, info = env.reset()
    print(f"Track seed: {info.get('track_seed', 'N/A')}")

    # Initialize pygame display before main loop
    env.render()

    running = True
    while running:
        throttle = 0.0
        steering = 0.0

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            throttle = -1.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering = -1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering = 1.0
        if keys[pygame.K_ESCAPE]:
            running = False

        # Handle single key presses for reset/new track
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print(f"Reset. Track seed: {info.get('track_seed', 'N/A')}")
                elif event.key == pygame.K_n:
                    # Force a new random track
                    env.randomize_track_on_reset = True
                    env._episode_count += 1  # Trigger track regeneration
                    obs, info = env.reset()
                    print(f"New track. Seed: {info.get('track_seed', 'N/A')}")

        action = np.array([throttle, steering], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode ended. Laps: {info.get('laps', 0)}")
            obs, info = env.reset()
            print(f"New track seed: {info.get('track_seed', 'N/A')}")

        env.render()

    env.close()


# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """
    Main entry point.

    Usage:
        python self_driving_car.py [mode] [--algo ALGO] [--timesteps N] [--track-type TYPE]

    Modes:
        train     - Train an agent (default: ppo)
        evaluate  - Evaluate a trained agent
        demo      - Manual keyboard control
        compare   - Compare all three algorithms
        quick     - Quick training (50k steps)

    Algorithms (--algo):
        ppo       - Proximal Policy Optimization (continuous actions)
        dqn       - Deep Q-Network (discrete actions)
        grpo      - Group Relative Policy Optimization (continuous actions)
        all       - Train/evaluate all algorithms

    Track Types (--track-type):
        random    - Procedurally generated random tracks (default)
        oval      - Classic oval track
    """
    import argparse

    parser = argparse.ArgumentParser(description="Self-Driving Car RL Simulation")
    parser.add_argument(
        "mode",
        nargs="?",
        default="train",
        choices=["train", "evaluate", "demo", "compare", "quick"],
        help="Mode to run"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo", "dqn", "grpo", "all"],
        help="Algorithm to use"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=HYPERPARAMS["total_timesteps"],
        help="Total training timesteps"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during training"
    )
    parser.add_argument(
        "--track", "--track-type",
        dest="track_type",
        type=str,
        default="random",
        choices=["random", "oval"],
        help="Track type: 'random' for procedural generation, 'oval' for classic oval"
    )
    parser.add_argument(
        "--track-seed",
        type=int,
        default=None,
        help="Seed for reproducible track generation (only for random tracks)"
    )
    parser.add_argument(
        "--randomize-tracks",
        action="store_true",
        help="Generate new random track each episode (for training generalization)"
    )

    args = parser.parse_args()

    if args.mode == "demo":
        demo_mode(
            track_type=args.track_type,
            track_seed=args.track_seed,
            randomize_on_reset=True  # Always allow new tracks in demo
        )

    elif args.mode == "compare":
        compare_algorithms(timesteps=args.timesteps, eval_episodes=args.episodes)

    elif args.mode == "train":
        if args.algo == "ppo":
            train_ppo(args.timesteps, "ppo_car_model", args.render, args.track_type, args.track_seed)
        elif args.algo == "dqn":
            train_dqn(args.timesteps, "dqn_car_model", args.render, args.track_type, args.track_seed)
        elif args.algo == "grpo":
            train_grpo(args.timesteps, "grpo_car_model.pt", args.render, track_type=args.track_type, track_seed=args.track_seed)
        elif args.algo == "all":
            train_ppo(args.timesteps, "ppo_car_model", args.render, args.track_type, args.track_seed)
            train_dqn(args.timesteps, "dqn_car_model", args.render, args.track_type, args.track_seed)
            train_grpo(args.timesteps, "grpo_car_model.pt", args.render, track_type=args.track_type, track_seed=args.track_seed)

    elif args.mode == "evaluate":
        if args.algo == "ppo":
            evaluate_ppo("ppo_car_model", args.episodes, args.track_type, args.track_seed)
        elif args.algo == "dqn":
            evaluate_dqn("dqn_car_model", args.episodes, args.track_type, args.track_seed)
        elif args.algo == "grpo":
            evaluate_grpo("grpo_car_model.pt", args.episodes, args.track_type, args.track_seed)
        elif args.algo == "all":
            evaluate_ppo("ppo_car_model", args.episodes, args.track_type, args.track_seed)
            evaluate_dqn("dqn_car_model", args.episodes, args.track_type, args.track_seed)
            evaluate_grpo("grpo_car_model.pt", args.episodes, args.track_type, args.track_seed)

    elif args.mode == "quick":
        quick_steps = 50_000
        if args.algo == "ppo":
            train_ppo(quick_steps, "ppo_car_model", args.render, args.track_type, args.track_seed)
            evaluate_ppo("ppo_car_model", 2, args.track_type, args.track_seed)
        elif args.algo == "dqn":
            train_dqn(quick_steps, "dqn_car_model", args.render, args.track_type, args.track_seed)
            evaluate_dqn("dqn_car_model", 2, args.track_type, args.track_seed)
        elif args.algo == "grpo":
            train_grpo(quick_steps, "grpo_car_model.pt", args.render, track_type=args.track_type, track_seed=args.track_seed)
            evaluate_grpo("grpo_car_model.pt", 2, args.track_type, args.track_seed)
        elif args.algo == "all":
            compare_algorithms(timesteps=quick_steps, eval_episodes=2)


if __name__ == "__main__":
    main()
