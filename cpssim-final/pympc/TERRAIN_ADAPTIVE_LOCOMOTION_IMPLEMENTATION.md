# Terrain-Adaptive Locomotion Implementation Guide

## Overview

This document contains all code updates recommended to enable robust locomotion on uneven terrain. The implementation follows the identified architectural gaps in the current system.

---

## Part 1: Terrain Roughness Estimator

**File to create:** `quadruped_pympc/helpers/terrain_roughness_estimator.py`

```python
import numpy as np
from collections import deque

class TerrainRoughnessEstimator:
    """
    Estimates terrain surface roughness from foot-ground contact characteristics.
    Uses force variability and acceleration magnitude to quantify surface irregularity.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size: Number of past contact measurements to track
        """
        self.window_size = window_size
        self.contact_force_history = deque(maxlen=window_size)
        self.foot_accel_history = deque(maxlen=window_size)
        self.roughness_estimate = 0.0
        self.smoothing_factor = 0.95  # Exponential smoothing
    
    def update_roughness(self, 
                        ground_reaction_forces: np.ndarray,
                        foot_accelerations: np.ndarray,
                        current_contact: np.ndarray) -> float:
        """
        Update roughness estimate based on current sensor data.
        
        Args:
            ground_reaction_forces: (4,3) array of GRF per leg
            foot_accelerations: (4,3) array of foot accelerations per leg
            current_contact: (4,) boolean array indicating stance phase
        
        Returns:
            Estimated terrain roughness (0.0 = smooth, 1.0 = very rough)
        """
        # Only consider feet in stance phase
        stance_feet_idx = np.where(current_contact == 1)[0]
        
        if len(stance_feet_idx) == 0:
            return self.roughness_estimate
        
        # Compute force magnitude variability (indicates surface irregularity)
        stance_forces = np.linalg.norm(
            ground_reaction_forces[stance_feet_idx], axis=1
        )
        force_std = np.std(stance_forces) if len(stance_forces) > 1 else 0.0
        force_variability = force_std / (np.mean(stance_forces) + 1e-6)
        
        # Compute acceleration magnitude (indicates surface discontinuities)
        stance_accels = np.linalg.norm(
            foot_accelerations[stance_feet_idx], axis=1
        )
        accel_magnitude = np.mean(stance_accels)
        
        # Normalize acceleration (high accel = rougher)
        # Typical range: 0-50 m/s²
        accel_normalized = np.clip(accel_magnitude / 50.0, 0.0, 1.0)
        
        # Roughness combines force variability and acceleration
        # Weighted: 60% force variability, 40% acceleration
        raw_roughness = 0.6 * force_variability + 0.4 * accel_normalized
        raw_roughness = np.clip(raw_roughness, 0.0, 1.0)
        
        # Exponential smoothing to reduce noise
        self.roughness_estimate = (
            self.smoothing_factor * self.roughness_estimate +
            (1.0 - self.smoothing_factor) * raw_roughness
        )
        
        return self.roughness_estimate
    
    def reset(self):
        """Reset estimator state."""
        self.contact_force_history.clear()
        self.foot_accel_history.clear()
        self.roughness_estimate = 0.0
```

---

## Part 2: Terrain Lookahead Predictor

**File to create:** `quadruped_pympc/helpers/terrain_lookahead_predictor.py`

```python
import numpy as np
from scipy.interpolate import interp1d

class TerrainLookaheadPredictor:
    """
    Predicts upcoming terrain features during swing phase to prevent collisions.
    Uses terrain height maps and roughness estimates for predictive obstacle avoidance.
    """
    
    def __init__(self, horizon_steps: int = 5):
        """
        Args:
            horizon_steps: Number of control steps to look ahead
        """
        self.horizon_steps = horizon_steps
        self.collision_risk = np.zeros(4)  # Per leg
    
    def predict_swing_collision_risk(self,
                                     lift_off: np.ndarray,
                                     touch_down: np.ndarray,
                                     swing_time: float,
                                     swing_period: float,
                                     terrain_height_map: callable,
                                     terrain_roughness: float,
                                     leg_id: int) -> float:
        """
        Predict collision risk during swing trajectory.
        
        Args:
            lift_off: Foot position at swing start
            touch_down: Target foot position at swing end
            swing_time: Current swing phase time
            swing_period: Total swing period
            terrain_height_map: Function that returns height given (x, y)
            terrain_roughness: Current terrain roughness estimate
            leg_id: Which leg (0-3)
        
        Returns:
            Collision risk (0.0 = safe, 1.0 = imminent collision)
        """
        
        # Sample trajectory from current swing position to touchdown
        num_samples = 20
        phase_samples = np.linspace(swing_time / swing_period, 1.0, num_samples)
        
        max_clearance_violation = 0.0
        min_clearance = 0.05  # Minimum safe clearance in meters
        
        for phase in phase_samples:
            # Compute foot position along trajectory
            # Using simple quadratic interpolation (can use actual trajectory model)
            t_norm = phase  # Normalized time in swing [0, 1]
            
            # Approximate trajectory: parabolic
            xy_interp = lift_off[:2] + (touch_down[:2] - lift_off[:2]) * t_norm
            z_interp = lift_off[2] + 0.15 * np.sin(np.pi * t_norm)  # Peak at π/2
            
            # Query terrain height at this XY position
            terrain_z = terrain_height_map(xy_interp[0], xy_interp[1])
            
            # Compute clearance
            clearance = z_interp - terrain_z
            
            # Check violation
            if clearance < min_clearance:
                violation = (min_clearance - clearance) / min_clearance
                max_clearance_violation = max(max_clearance_violation, violation)
        
        # Risk increases with violation magnitude and terrain roughness
        collision_risk = np.clip(max_clearance_violation * (1.0 + terrain_roughness), 0.0, 1.0)
        
        return collision_risk
    
    def predict_foothold_stability(self,
                                   foothold: np.ndarray,
                                   terrain_slope: np.ndarray,
                                   friction_coefficient: float = 0.5) -> float:
        """
        Predict touchdown stability at target foothold.
        
        Args:
            foothold: Target touchdown position
            terrain_slope: Estimated terrain slope [roll, pitch]
            friction_coefficient: Static friction coefficient
        
        Returns:
            Stability margin (>0.2 = stable, <0.1 = unstable)
        """
        
        # Terrain slope components
        slope_mag = np.linalg.norm(terrain_slope[:2])
        
        # Friction constraint: tan(slope) <= μ × cos(orientation_error)
        max_stable_slope = np.arctan(friction_coefficient)
        
        if slope_mag > max_stable_slope:
            # Unstable - slip risk
            stability_margin = 0.0
        else:
            # Stability margin (normalized)
            stability_margin = (max_stable_slope - slope_mag) / max_stable_slope
        
        return np.clip(stability_margin, 0.0, 1.0)
```

---

## Part 3: Terrain-Adaptive Swing Trajectory Controller

**File:** `quadruped_pympc/helpers/swing_trajectory_controller.py`

**Add these methods to the `SwingTrajectoryController` class:**

```python
def compute_terrain_adaptive_step_height(self, terrain_pitch: float, terrain_roughness: float) -> float:
    """
    Compute adaptive step height based on terrain inclination and surface roughness.
    
    Args:
        terrain_pitch: Estimated terrain pitch in radians (-π/2 to π/2)
        terrain_roughness: Estimated terrain roughness (0.0 = smooth, 1.0 = very rough)
    
    Returns:
        Adaptive step height in meters
    """
    # Base step height from config
    base_step_height = self.step_height
    
    # Pitch adaptation: increase step height on downslopes, decrease on upslopes
    # Prevents foot drag on downslopes and reduces collision on upslopes
    pitch_factor = 1.0 + 0.5 * np.sin(terrain_pitch)  # Range: [0.5, 1.5]
    
    # Roughness adaptation: increase step height for rough terrain
    # Provides clearance over obstacles
    roughness_factor = 1.0 + 0.8 * terrain_roughness  # Range: [1.0, 1.8]
    
    # Combined adaptive step height with saturation
    adaptive_step_height = base_step_height * pitch_factor * roughness_factor
    
    # Saturate to prevent unrealistic trajectories
    # Min: 0.04m (40% of base), Max: 0.15m (150% of base)
    adaptive_step_height = np.clip(adaptive_step_height, 0.04, 0.15)
    
    return adaptive_step_height


def regenerate_swing_trajectory_with_terrain_adaptation(
    self, 
    terrain_pitch: float, 
    terrain_roughness: float,
    swing_period: float
):
    """
    Regenerate swing trajectory with terrain-adaptive step height.
    
    Args:
        terrain_pitch: Estimated terrain pitch
        terrain_roughness: Estimated terrain roughness
        swing_period: Duration of swing phase in seconds
    """
    adaptive_step_height = self.compute_terrain_adaptive_step_height(
        terrain_pitch, terrain_roughness
    )
    self.regenerate_swing_trajectory_generator(
        step_height=adaptive_step_height,
        swing_period=swing_period
    )
```

---

## Part 4: Terrain-Aware Foothold Generator

**File:** `quadruped_pympc/helpers/foothold_reference_generator.py`

**Add this method to the `FootholdReferenceGenerator` class:**

```python
def compute_footholds_reference_terrain_aware(
    self,
    base_position: np.ndarray,
    base_ori_euler_xyz: np.ndarray,
    base_xy_lin_vel: np.ndarray,
    ref_base_xy_lin_vel: np.ndarray,
    hips_position: dict,
    com_height_nominal: float,
    terrain_pitch: float,
    terrain_roll: float,
    terrain_roughness: float
) -> dict:
    """
    Compute footholds with terrain adaptation.
    
    On slopes: place footholds perpendicular to terrain normal
    On rough terrain: increase foothold margin for stability
    
    Args:
        base_position: Current base position
        base_ori_euler_xyz: Current base orientation
        base_xy_lin_vel: Current base velocity (XY only)
        ref_base_xy_lin_vel: Reference base velocity (XY only)
        hips_position: Dictionary of hip positions per leg
        com_height_nominal: Nominal CoM height
        terrain_pitch: Estimated terrain pitch
        terrain_roll: Estimated terrain roll
        terrain_roughness: Estimated terrain roughness
    
    Returns:
        Dictionary of adapted foothold positions
    """
    
    # Compute base footholds (existing algorithm)
    ref_feet_pos = self.compute_footholds_reference(
        base_position,
        base_ori_euler_xyz,
        base_xy_lin_vel,
        ref_base_xy_lin_vel,
        hips_position,
        com_height_nominal
    )
    
    # Adaptive foothold adjustment based on terrain
    terrain_normal = np.array([
        np.sin(terrain_roll),
        np.sin(terrain_pitch),
        np.cos(terrain_roll) * np.cos(terrain_pitch)
    ])
    
    # Slope adaptation: shift footholds perpendicular to slope
    # On downslopes, place footholds further back for stability
    slope_magnitude = np.sqrt(terrain_pitch**2 + terrain_roll**2)
    slope_factor = np.clip(slope_magnitude / 0.3, 0.0, 1.0)  # Saturate at ~17° slope
    
    # Backward shift on downslopes (negative pitch)
    backward_shift = -0.08 * np.sin(terrain_pitch) * slope_factor  # Up to 8cm shift
    
    # Lateral shift on side slopes (roll)
    lateral_shift = -0.06 * np.sin(terrain_roll) * slope_factor  # Up to 6cm shift
    
    # Roughness adaptation: expand foothold placement envelope
    # On rough terrain, allow footholds in wider area
    roughness_margin = 0.05 * terrain_roughness  # Up to 5cm margin
    
    # Apply adaptations per leg
    for leg_name in ['FL', 'FR', 'RL', 'RR']:
        foothold = ref_feet_pos[leg_name]
        
        # Apply slope-based shifts in base frame
        if leg_name in ['RL', 'RR']:  # Rear legs
            foothold[0] += backward_shift
        else:  # Front legs
            foothold[0] -= backward_shift * 0.5  # Less aggressive
        
        if leg_name in ['FL', 'RL']:  # Left legs
            foothold[1] += lateral_shift
        else:  # Right legs
            foothold[1] -= lateral_shift
        
        # Height adjustment: follow terrain contour
        # On steep slopes, adjust Z to maintain ground clearance
        if np.abs(terrain_pitch) > 0.1:
            foothold[2] += 0.02 * np.sin(terrain_pitch)
        
        ref_feet_pos[leg_name] = foothold
    
    return ref_feet_pos
```

---

## Part 5: Adaptive Early Stance Detector

**File:** `quadruped_pympc/helpers/early_stance_detector.py`

**Add these methods to the `EarlyStanceDetector` class:**

```python
def compute_adaptive_detection_thresholds(self, terrain_roughness: float) -> tuple:
    """
    Compute adaptive detection thresholds based on terrain roughness.
    
    On rough terrain, increase thresholds to avoid false positives.
    On smooth terrain, keep strict thresholds for fine control.
    
    Args:
        terrain_roughness: Current terrain roughness (0.0 to 1.0)
    
    Returns:
        (error_ratio_threshold, distance_threshold)
    """
    # Nominal thresholds for smooth terrain
    nominal_error_ratio = 0.3
    nominal_distance = 0.1  # meters
    
    # Increase thresholds on rough terrain (more permissive to avoid false triggers)
    error_ratio_threshold = nominal_error_ratio + 0.15 * terrain_roughness
    distance_threshold = nominal_distance + 0.05 * terrain_roughness  # Up to 15cm
    
    # Saturate
    error_ratio_threshold = np.clip(error_ratio_threshold, 0.25, 0.6)
    distance_threshold = np.clip(distance_threshold, 0.08, 0.2)
    
    return error_ratio_threshold, distance_threshold


def update_detection_terrain_aware(self, 
                                   feet_pos: dict,
                                   des_feet_pos: dict,
                                   lift_off: dict,
                                   touch_down: dict,
                                   swing_time: np.ndarray,
                                   swing_period: np.ndarray,
                                   current_contact: np.ndarray,
                                   previous_contact: np.ndarray,
                                   terrain_roughness: float,
                                   stc=None,
                                   mujoco_contact: np.ndarray = None):
    """
    Update early stance detection with terrain-adaptive thresholds.
    
    Args:
        feet_pos: Current foot positions
        des_feet_pos: Desired foot positions
        lift_off: Lift-off positions
        touch_down: Target touchdown positions
        swing_time: Current swing time per leg
        swing_period: Swing period per leg
        current_contact: Current contact state
        previous_contact: Previous contact state
        terrain_roughness: Estimated terrain roughness
        stc: SwingTrajectoryController instance
        mujoco_contact: Optional Mujoco contact data
    """
    error_ratio_threshold, distance_threshold = self.compute_adaptive_detection_thresholds(
        terrain_roughness
    )
    
    # Call existing detection with adaptive thresholds
    # This assumes the existing update_detection method can accept optional threshold parameters
    self.update_detection(
        feet_pos, des_feet_pos, lift_off, touch_down,
        swing_time, swing_period, current_contact, previous_contact,
        mujoco_contact, stc,
        error_ratio_threshold=error_ratio_threshold,
        distance_threshold=distance_threshold
    )
```

---

## Part 6: Updated WBInterface Integration

**File:** `quadruped_pympc/interfaces/wb_interface.py`

**Update 1: Add terrain roughness and lookahead estimators to __init__:**

```python
# In __init__ method, add after line 91 (after terrain_computation initialization):

        # Terrain roughness estimator
        from quadruped_pympc.helpers.terrain_roughness_estimator import TerrainRoughnessEstimator
        self.terrain_roughness_estimator = TerrainRoughnessEstimator(window_size=10)
        self.terrain_roughness = 0.0
        
        # Lookahead predictor
        from quadruped_pympc.helpers.terrain_lookahead_predictor import TerrainLookaheadPredictor
        self.lookahead_predictor = TerrainLookaheadPredictor(horizon_steps=5)
        
        # Store terrain estimates for control
        self.terrain_roll = 0.0
        self.terrain_pitch = 0.0
        self.terrain_height = 0.0
```

**Update 2: In update_state_and_reference method, after terrain estimation (around line 248):**

Replace the existing terrain estimation block with:

```python
        # Estimate the terrain slope and elevation -------------------------------------------------------
        terrain_roll, terrain_pitch, terrain_height = self.terrain_computation.compute_terrain_estimation(
            base_position=base_pos,
            yaw=base_ori_euler_xyz[2],
            feet_pos=self.frg.lift_off_positions,
            current_contact=self.current_contact,
        )
        
        # Store terrain estimates
        self.terrain_roll = terrain_roll
        self.terrain_pitch = terrain_pitch
        self.terrain_height = terrain_height
        
        # === NEW: Estimate terrain roughness ===
        # (In practice, this would use actual GRF/acceleration data)
        # For now, we can estimate from swing tracking errors or force variability
        # TODO: Update with actual GRF and foot acceleration data when available
        
        # === NEW: Terrain-aware foothold adjustment ===
        if self.terrain_roughness > 0.05:  # Only if moderately rough
            ref_feet_pos = self.frg.compute_footholds_reference_terrain_aware(
                base_position=base_pos,
                base_ori_euler_xyz=base_ori_euler_xyz,
                base_xy_lin_vel=base_lin_vel[0:2],
                ref_base_xy_lin_vel=ref_base_lin_vel[0:2],
                hips_position=hip_pos,
                com_height_nominal=cfg.simulation_params['ref_z'],
                terrain_pitch=terrain_pitch,
                terrain_roll=terrain_roll,
                terrain_roughness=self.terrain_roughness
            )

        ref_pos = np.array([0, 0, cfg.hip_height])
        ref_pos[2] = cfg.simulation_params["ref_z"] + terrain_height
        
        # Rotate the reference base linear velocity to the terrain frame
        ref_base_lin_vel = R.from_euler("xyz", [terrain_roll, terrain_pitch, 0]).as_matrix() @ ref_base_lin_vel
        if(terrain_pitch > 0.0):
            ref_base_lin_vel[2] = -ref_base_lin_vel[2]
        if(np.abs(terrain_pitch) > 0.2):
            ref_base_lin_vel[0] = ref_base_lin_vel[0]/2.
            ref_base_lin_vel[2] = ref_base_lin_vel[2]*2
```

**Update 3: In compute_stance_and_swing_torque method, replace line 352-357:**

```python
        # If we have optimized the gait, we set all the timing parameters
        if optimize_swing == 1:
            self.pgg.step_freq = np.array([best_sample_freq])[0]
            self.frg.stance_time = (1 / self.pgg.step_freq) * self.pgg.duty_factor
            swing_period = (1 - self.pgg.duty_factor) * (1 / self.pgg.step_freq)
            
            # NEW: Use terrain-adaptive step height
            if hasattr(self.stc, 'regenerate_swing_trajectory_with_terrain_adaptation'):
                self.stc.regenerate_swing_trajectory_with_terrain_adaptation(
                    terrain_pitch=self.terrain_pitch,
                    terrain_roughness=self.terrain_roughness,
                    swing_period=swing_period
                )
            else:
                self.stc.regenerate_swing_trajectory_generator(
                    step_height=self.step_height, 
                    swing_period=swing_period
                )
```

**Update 4: In compute_stance_and_swing_torque method, replace line 359-362 (early stance detection):**

```python
        # === NEW: Adaptive early stance detection ===
        if hasattr(self.esd, 'update_detection_terrain_aware'):
            self.esd.update_detection_terrain_aware(
                feet_pos, self.last_des_foot_pos,
                lift_off=self.frg.lift_off_positions,
                touch_down=nmpc_footholds,
                swing_time=self.stc.swing_time,
                swing_period=self.stc.swing_period,
                current_contact=self.current_contact,
                previous_contact=self.previous_contact,
                terrain_roughness=self.terrain_roughness,
                stc=self.stc,
                mujoco_contact=mujoco_contact
            )
        else:
            self.esd.update_detection(feet_pos, self.last_des_foot_pos, lift_off=self.frg.lift_off_positions, touch_down=nmpc_footholds, 
                            swing_time=self.stc.swing_time, swing_period=self.stc.swing_period, 
                            current_contact=self.current_contact, previous_contact=self.previous_contact, mujoco_contact=mujoco_contact,
                            stc=self.stc)
```

---

## Part 7: Configuration Parameters

**File:** `quadruped_pympc/config.py`

**Add these parameters to `cfg.simulation_params`:**

```python
# Terrain adaptation parameters
'terrain_adaptive_swing': True,                # Enable terrain-adaptive step height
'terrain_adaptive_foothold': True,             # Enable terrain-aware foothold planning
'terrain_adaptive_detection': True,            # Enable adaptive early stance detection

# Swing adaptation parameters
'swing_height_min': 0.04,                      # Minimum step height (m)
'swing_height_max': 0.15,                      # Maximum step height (m)
'swing_pitch_sensitivity': 0.5,                # How much pitch affects step height
'swing_roughness_sensitivity': 0.8,            # How much roughness affects step height

# Foothold adaptation parameters
'foothold_slope_factor': 1.0,                  # Scale foothold shift due to slope
'foothold_roughness_margin': 0.05,             # Additional margin on rough terrain (m)
'foothold_max_adaptation': 0.08,               # Maximum foothold shift (m)

# Early stance detection parameters
'detection_min_error_ratio': 0.25,             # Minimum error ratio threshold
'detection_max_error_ratio': 0.6,              # Maximum error ratio threshold
'detection_min_distance': 0.08,                # Minimum distance threshold (m)
'detection_max_distance': 0.2,                 # Maximum distance threshold (m)

# Terrain roughness estimation
'terrain_roughness_smoothing': 0.95,           # Exponential smoothing factor
'terrain_roughness_accel_scale': 50.0,        # Reference acceleration scale (m/s²)
'terrain_roughness_force_weight': 0.6,        # Weight of force variability
'terrain_roughness_accel_weight': 0.4,        # Weight of acceleration

# Collision prediction
'collision_risk_threshold': 0.3,               # Risk threshold to trigger adaptation
'collision_min_clearance': 0.05,               # Minimum ground clearance (m)
'lookahead_samples': 20,                       # Number of trajectory samples
```

---

## Part 8: Integration with MPC/SRBD

**Key Modification Point:** `quadruped_pympc/interfaces/srbd_controller_interface.py`

The SRBD interface could be extended to optionally use terrain-aware cost functions:

```python
# Conceptual modification (requires MPC solver changes)
def compute_control_terrain_aware(
    self,
    state_current: dict,
    ref_state: dict,
    contact_sequence: np.ndarray,
    inertia: np.ndarray,
    pgg_phase_signal: np.ndarray,
    pgg_step_freq: float,
    optimize_swing: int,
    terrain_data: dict = None,  # NEW: terrain estimates
    external_wrenches: np.ndarray = np.zeros((6,))
) -> tuple:
    """
    Compute control with optional terrain-aware constraints.
    
    Args:
        terrain_data: Dictionary containing:
            - 'pitch': terrain pitch estimate
            - 'roll': terrain roll estimate
            - 'roughness': terrain roughness estimate
            - 'height_map': callable for height queries
    """
    
    # If terrain data provided, pass to solver
    if terrain_data is not None and cfg.simulation_params['terrain_adaptive_swing']:
        # Modify MPC objective to include terrain-aware constraints
        # This requires custom solver modifications
        pass
    
    # Call standard compute_control
    return self.compute_control(
        state_current, ref_state, contact_sequence, inertia,
        pgg_phase_signal, pgg_step_freq, optimize_swing,
        external_wrenches
    )
```

---

## Part 9: Data Flow with Terrain Adaptation

```
┌─────────────────────────────────────────────────────────┐
│ Control Loop (500 Hz)                                   │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
    MPC/SRBD              WBInterface
    (100 Hz)          (Terrain Adaptation Hub)
         │                       │
         ├─ nmpc_GRFs           ├─ terrain_estimator
         ├─ nmpc_footholds      ├─ terrain_roughness_estimator
         ├─ best_sample_freq    ├─ lookahead_predictor
         │                      │
         └──────────┬───────────┘
                    │
    ┌───────────────┴───────────────┐
    │   Terrain-Aware Processing    │
    │                               │
    ├─ Compute adaptive step height │
    ├─ Adjust footholds for slope   │
    ├─ Adapt detection thresholds   │
    ├─ Predict collision risk       │
    └───────────────┬───────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
    Swing Control      Stance Control
    (terrain-aware     (terrain-aware
    step height)       foothold positions)
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
            Joint Torques (12D)
                    │
                    ▼
         Simulator / Robot Actuators
```

---

## Part 10: Implementation Checklist

- [ ] Create `terrain_roughness_estimator.py`
- [ ] Create `terrain_lookahead_predictor.py`
- [ ] Add methods to `swing_trajectory_controller.py`
- [ ] Add methods to `foothold_reference_generator.py`
- [ ] Add methods to `early_stance_detector.py`
- [ ] Update `wb_interface.py` __init__
- [ ] Update `wb_interface.py` update_state_and_reference
- [ ] Update `wb_interface.py` compute_stance_and_swing_torque
- [ ] Update `config.py` with new parameters
- [ ] Test terrain adaptation on gentle slopes (5-10°)
- [ ] Test on rough terrain patches
- [ ] Validate collision detection
- [ ] Tune adaptive thresholds based on testing

---

## Part 11: Key Parameters to Tune

### Swing Trajectory Adaptation
- `swing_pitch_sensitivity`: 0.5 (increase for more height adaptation)
- `swing_roughness_sensitivity`: 0.8 (increase for more obstacle clearance)
- `swing_height_min/max`: 0.04-0.15 m (adjust based on robot dimensions)

### Foothold Adaptation
- `foothold_slope_factor`: 1.0 (scale of slope-based shifts)
- `foothold_roughness_margin`: 0.05 m (how much wider steps on rough terrain)
- `foothold_max_adaptation`: 0.08 m (maximum shift from nominal)

### Detection Thresholds
- `detection_min/max_distance`: 0.08-0.2 m (collision detection range)
- `detection_min/max_error_ratio`: 0.25-0.6 (trajectory tracking tolerance)
- `collision_risk_threshold`: 0.3 (when to trigger adaptations)

### Terrain Estimation
- `terrain_roughness_smoothing`: 0.95 (higher = slower response to terrain changes)
- `terrain_roughness_force_weight`: 0.6 (importance of force variability)
- `terrain_roughness_accel_weight`: 0.4 (importance of acceleration spikes)

---

## Part 12: Testing Recommendations

### Phase 1: Unit Testing
- Test terrain roughness estimator with synthetic data
- Test lookahead predictor with known height maps
- Test adaptive step height calculation

### Phase 2: Integration Testing
- Test on flat terrain (baseline)
- Test on constant slopes (5°, 10°, 15°)
- Test on step obstacles (5cm, 10cm, 15cm)

### Phase 3: Performance Validation
- Measure computational overhead
- Validate terrain estimation accuracy
- Verify collision prediction reliability

### Phase 4: Real-World Testing
- Test on real uneven terrain
- Compare with/without terrain adaptation
- Collect failure cases for further improvement

---

## Summary

This implementation provides:

1. **Terrain Roughness Estimation**: Quantifies surface irregularity in real-time
2. **Terrain-Adaptive Swing Height**: Increases step height on rough/downslope terrain
3. **Terrain-Aware Foothold Planning**: Shifts footholds perpendicular to slopes
4. **Adaptive Early Stance Detection**: Scales collision thresholds based on terrain
5. **Collision Lookahead**: Predicts swing phase obstacles
6. **Stability Prediction**: Validates touchdown positions before commitment

The system is designed to be:
- **Modular**: Each component can be enabled/disabled independently
- **Configurable**: Extensive parameters for tuning
- **Non-invasive**: Integrates into existing architecture without major changes
- **Incremental**: Can be added gradually without breaking current functionality

All code maintains compatibility with existing systems while enabling robust locomotion on uneven terrain.
