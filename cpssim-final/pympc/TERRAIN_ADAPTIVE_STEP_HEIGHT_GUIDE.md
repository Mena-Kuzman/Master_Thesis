# Terrain-Adaptive Step Height Implementation Guide

## Overview
Instead of using a hardcoded step height, the algorithm adapts the step height based on:
1. **Terrain slope (pitch)**: Affects foot clearance on inclines
2. **Terrain roughness**: Affects obstacle avoidance
3. **Velocity**: Higher speeds may require different clearance

---

## Mathematical Relationships

### 1. Base Step Height
```
h_base = 0.2 × h_hip  (from config.py)
```
Current value: ~0.056m (for h_hip = 0.28m)

### 2. Terrain Pitch Adaptation
```
pitch_factor = 1.0 + 0.5 × sin(θ_pitch)

where:
  θ_pitch = terrain pitch angle in radians
  Range: [0.5, 1.5]
  
Interpretation:
  - θ_pitch = -45° → sin(-π/4) ≈ -0.707 → pitch_factor ≈ 0.65 (downslope: reduce height)
  - θ_pitch = 0°   → sin(0) = 0      → pitch_factor = 1.0  (flat: baseline)
  - θ_pitch = +45° → sin(π/4) ≈ +0.707 → pitch_factor ≈ 1.35 (upslope: increase height)
```

**Rationale:**
- On **downslopes**: Reduce step height to prevent foot drag and maintain contact
- On **upslopes**: Increase step height for better clearance over uneven terrain
- On **flat terrain**: Use baseline height

### 3. Terrain Roughness Adaptation
```
roughness_factor = 1.0 + 0.8 × σ_roughness

where:
  σ_roughness = estimated terrain roughness (0.0 = smooth, 1.0 = very rough)
  Range: [1.0, 1.8]
  
Interpretation:
  - σ_roughness = 0.0 → roughness_factor = 1.0 (smooth: baseline)
  - σ_roughness = 0.5 → roughness_factor = 1.4 (moderate: +40%)
  - σ_roughness = 1.0 → roughness_factor = 1.8 (rough: +80%)
```

**Rationale:**
- Rough terrain has unpredictable obstacles
- Increased step height provides clearance margins over obstacles
- Prevents collision with hidden bumps or rocks

### 4. Velocity-Dependent Adaptation (Optional)
```
velocity_factor = 1.0 + 0.3 × (v_commanded / v_max)

where:
  v_commanded = command velocity m/s
  v_max = maximum design velocity (e.g., 2.0 m/s)
  Range: [1.0, 1.3]
  
Interpretation:
  - v = 0 m/s   → velocity_factor = 1.0 (standing: baseline)
  - v = 1 m/s   → velocity_factor = 1.15 (walking: +15%)
  - v = 2 m/s   → velocity_factor = 1.3 (running: +30%)
```

**Rationale:**
- Higher speeds require faster ground clearance to avoid mid-swing contact
- Increased margins for dynamic stability
- Prevents toe-drag during high-speed gaits

### 5. Combined Adaptive Step Height
```
h_adaptive = h_base × pitch_factor × roughness_factor × velocity_factor

Saturation constraints:
  h_min = 0.04 m  (minimum clearance)
  h_max = 0.15 m  (maximum realistic trajectory)
  
Final:
  h_adaptive = clip(h_adaptive, h_min, h_max)
```

---

## Data Flow

```
Current State
    ↓
Terrain Estimator (terrain_estimator.py)
    ├─ Estimates pitch: θ_pitch
    ├─ Estimates roll: θ_roll
    └─ Estimates height: z_terrain
    ↓
Terrain Roughness Estimator (to be added)
    └─ Estimates roughness: σ_roughness
    ↓
Velocity Command
    └─ Gets v_commanded
    ↓
Compute Terrain-Adaptive Step Height
    ├─ pitch_factor = 1.0 + 0.5 × sin(θ_pitch)
    ├─ roughness_factor = 1.0 + 0.8 × σ_roughness
    ├─ velocity_factor = 1.0 + 0.3 × (v_cmd / v_max)
    ├─ h_adaptive = h_base × pitch_factor × roughness_factor × velocity_factor
    └─ h_final = clip(h_adaptive, 0.04, 0.15)
    ↓
SwingTrajectoryController
    └─ regenerate_swing_trajectory_generator(h_final, swing_period)
    ↓
Swing Trajectory Generator (scipy or explicit)
    └─ Generates swing trajectory with height h_final
    ↓
Swing Trajectory Controller
    └─ Tracks trajectory with computed forces
```

---

## Current Implementation Status

### Already Implemented ✅
- **Terrain Estimator** (`helpers/terrain_estimator.py`)
  - Computes pitch, roll, height from foot positions
  - Integrated in `wb_interface.py` (lines 248-265)
  - Returns: `terrain_roll`, `terrain_pitch`, `terrain_height`

### Missing / To Be Added ❌
1. **Terrain Roughness Estimator**
   - Could estimate from:
     - GRF variance during stance
     - Swing tracking error variance
     - IMU acceleration variance

2. **Adaptive Step Height Computation**
   - Method: `compute_terrain_adaptive_step_height()`
   - Location: `swing_trajectory_controller.py`
   - Called in: `wb_interface.py` update loop

3. **Integration with Adaptive Gait**
   - Combine with existing adaptive gait (period/duty factor)
   - Ensure velocity-dependent adaptation doesn't conflict

---

## Implementation Steps

### Step 1: Add Configuration Parameters
**File:** `quadruped_pympc/config.py`

Add to `simulation_params`:
```python
# Terrain-adaptive step height
'terrain_adaptive_step_height_enabled': True,
'swing_height_base': 0.2 * hip_height,  # Base step height
'swing_height_min': 0.04,               # Minimum (m)
'swing_height_max': 0.15,               # Maximum (m)
'swing_pitch_sensitivity': 0.5,         # Pitch factor coefficient
'swing_roughness_sensitivity': 0.8,     # Roughness factor coefficient
'swing_velocity_sensitivity': 0.3,      # Velocity factor coefficient
```

### Step 2: Implement Roughness Estimator
**File:** `quadruped_pympc/helpers/terrain_roughness_estimator.py` (new)

```python
class TerrainRoughnessEstimator:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.grf_history = collections.deque(maxlen=window_size)
        self.roughness = 0.0
    
    def update(self, grf: np.ndarray, contact: np.ndarray):
        """Estimate roughness from GRF variance during stance"""
        if np.any(contact):  # Only update during stance
            self.grf_history.append(np.linalg.norm(grf))
        
        if len(self.grf_history) == self.window_size:
            # Normalize variance to [0, 1]
            variance = np.var(list(self.grf_history))
            self.roughness = np.clip(variance / 1000, 0, 1)
        
        return self.roughness
```

### Step 3: Add Step Height Computation
**File:** `quadruped_pympc/helpers/swing_trajectory_controller.py`

Add method:
```python
def compute_terrain_adaptive_step_height(self, 
                                        terrain_pitch: float,
                                        terrain_roughness: float,
                                        velocity: float,
                                        velocity_max: float = 2.0) -> float:
    """Compute adaptive step height based on terrain and velocity"""
    
    # Base height
    h_base = self.step_height
    
    # Pitch factor: increase on upslopes
    pitch_factor = 1.0 + 0.5 * np.sin(terrain_pitch)
    
    # Roughness factor: increase on rough terrain
    roughness_factor = 1.0 + 0.8 * terrain_roughness
    
    # Velocity factor: increase at higher speeds
    velocity_factor = 1.0 + 0.3 * (velocity / velocity_max)
    
    # Combined adaptation with saturation
    h_adaptive = h_base * pitch_factor * roughness_factor * velocity_factor
    
    # Clip to safe range
    h_min = cfg.simulation_params.get('swing_height_min', 0.04)
    h_max = cfg.simulation_params.get('swing_height_max', 0.15)
    h_final = np.clip(h_adaptive, h_min, h_max)
    
    return h_final
```

### Step 4: Integrate with WBInterface
**File:** `quadruped_pympc/interfaces/wb_interface.py`

In `update_state_and_reference()` method, after line 265 (terrain estimation):

```python
# Estimate terrain roughness
if not hasattr(self, 'roughness_estimator'):
    from quadruped_pympc.helpers.terrain_roughness_estimator import TerrainRoughnessEstimator
    self.roughness_estimator = TerrainRoughnessEstimator()

terrain_roughness = self.roughness_estimator.update(
    grf_measured,  # Current estimated GRF
    self.current_contact
)

# Compute adaptive step height
if cfg.simulation_params.get('terrain_adaptive_step_height_enabled', False):
    velocity = float(np.linalg.norm(base_lin_vel[0:2]))
    adaptive_step_height = self.stc.compute_terrain_adaptive_step_height(
        terrain_pitch=terrain_pitch,
        terrain_roughness=terrain_roughness,
        velocity=velocity,
        velocity_max=2.0
    )
    
    # Regenerate swing with new height
    swing_period = (1.0 - self.pgg.duty_factor) / self.pgg.step_freq
    self.stc.regenerate_swing_trajectory_generator(
        step_height=adaptive_step_height,
        swing_period=swing_period
    )
else:
    # Use fixed step height
    self.stc.regenerate_swing_trajectory_generator(
        step_height=self.step_height,
        swing_period=swing_period
    )
```

---

## Mathematical Intuition

### Why Sine Function for Pitch?
```
The swing trajectory is roughly sinusoidal in height:
  z(t) = z_0 + h × sin(π × t / T)

On an upslope, the ground reference is tilted:
  z_ground(x) = x × tan(θ)

The sinusoidal adaptation compensates for this tilt:
  - When θ > 0 (upslope): Increase h to maintain clearance
  - When θ < 0 (downslope): Decrease h to prevent foot drag
```

### Why Linear for Roughness?
```
Roughness adds random obstacles:
  z_actual(x, y) = z_nominal(x, y) + noise(σ)

Clearance margin scales linearly with expected obstacle size:
  h_margin = k × σ

Hence: roughness_factor = 1 + 0.8 × σ
```

### Why Scaling Factors are < 1?
```
Conservative design: factors multiply step height
  - Pitch: 0.5 (±25% variation from base)
  - Roughness: 0.8 (+80% max increase)
  - Velocity: 0.3 (+30% max increase)

Prevents extreme trajectories that might:
  1. Hit joint limits
  2. Exceed torque limits
  3. Create infeasible kinematics
```

---

## Testing Strategy

### 1. Unit Tests
- Test `compute_terrain_adaptive_step_height()` with known inputs
- Verify saturation constraints (0.04 ≤ h ≤ 0.15)
- Check factor calculations independently

### 2. Simulation Tests
- **Flat terrain**: h_adaptive should ≈ h_base (all factors ≈ 1.0)
- **Upslope (+15°)**: h_adaptive should increase ~25%
- **Downslope (-15°)**: h_adaptive should decrease ~25%
- **Rough terrain** (σ = 0.5): h_adaptive should increase ~40%
- **High velocity** (2 m/s): h_adaptive should increase ~30%

### 3. Integration Tests
- Run full locomotion on varied terrain
- Monitor:
  - Step height over time
  - Swing clearance margins
  - Control stability
  - Tracking performance

---

## Expected Benefits

1. **Better stability on slopes**: Adaptive height prevents foot drag
2. **Improved rough terrain traversal**: Increased clearance over obstacles
3. **Higher speed capability**: Velocity-dependent adaptation maintains safety
4. **Reduced collision events**: Early stance detector triggers less often
5. **Smoother velocity transitions**: Continuous adaptation vs. discrete gait switching

---

## References

- `quadruped_pympc/helpers/terrain_estimator.py`: Terrain slope estimation
- `quadruped_pympc/helpers/swing_trajectory_controller.py`: Swing control
- `quadruped_pympc/helpers/swing_generators/`: Trajectory generation
- `TERRAIN_ADAPTIVE_LOCOMOTION_IMPLEMENTATION.md`: Full implementation guide
