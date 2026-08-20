# DFKI Quadruped Gait Sequencer Implementation Analysis

## Overview

The DFKI quadruped controller implements three distinct gait sequencing approaches integrated with their MPC-based walking controller. All three approaches work with the same core contact sequence data structure and integrate with foothold planning via Raibert-based algorithms.

---

## Gait Sequencer Architectures

### 1. **SimpleGaitSequencer** (Static/Phase-Based)

**Type:** Static, periodic gait sequencer with simple phase calculation

**Key Characteristics:**
- Uses a fixed gait profile with constant period `T`, duty factor per leg, and phase offsets
- Phase calculation: `phase = (time_in_phase / T) % 1.0`
- Contact determined by: `contact[i] = phase[i] < duty_factor[i]` for each leg
- **Early contact detection** (optional): Detects foot contact before planned contact time during swing phase to extend stance phase

**Parameters:**
- `period` (T): Total gait cycle time (e.g., 0.5s for trot)
- `duty_factor`: Fraction of cycle each leg is in contact (e.g., 0.5 = 50% stance)
- `phase_offset`: Relative phase of each leg (e.g., trot = [0.0, 0.5, 0.5, 0.0])

**Contact Sequence Generation:**
```cpp
for each timestep i in GAIT_SEQUENCE_SIZE:
    for each leg:
        leg_phase = (global_phase + i*dt/T - phase_offset) mod 1.0
        contact[i][leg] = leg_phase < duty_factor[leg]
```

**Integration Points:**
- Calls `RaibertFootStepPlanner` to compute foot positions
- Uses `MPCTrajectoryPlanner` for desired trajectory generation
- Early contact detected from actual foot sensor states

---

### 2. **BioInspiredGaitSequencer** (Adaptive Gait Switching)

**Type:** Froude-number-based adaptive gait sequencer with smooth transitions

**Key Characteristics:**
- Dynamically switches between predefined gaits based on locomotion speed
- Uses **Froude number** to determine appropriate gait: $Fr = \frac{v^2}{g \cdot h}$
- Implements **smooth gait transitions** using interpolation weights
- Maintains a database of predefined gaits with Froude number bounds

**Gait Database (BioGaitDatabase):**
```
Standing:        Fr < 0.0009
Static Walk:     0.0009 ≤ Fr < 0.0024
Walking Trot:    0.0024 ≤ Fr < 0.1517
Trot:            0.1517 ≤ Fr < 0.74
Flying Trot:     0.74 ≤ Fr
```

**Transition Mechanism:**

When Froude number indicates a gait change, implements smooth interpolation:

$$\text{period} = w_1 \cdot T_{\text{old}} + w_2 \cdot T_{\text{new}}$$
$$\text{duty\_factor}[i] = w_1 \cdot df_{\text{old}}[i] + w_2 \cdot df_{\text{new}}[i]$$
$$\text{phase\_offset}[i] = \text{offset}_{\text{old}}[i] + (\text{phase\_error}) \cdot w_2$$

Where:
- $w_2 = \min\left(\frac{n_{\text{trans}} / \text{dt}}{D_{\text{trans}}}, 1.0\right)$ (transition progress)
- $w_1 = 1.0 - w_2$ (complementary weight)
- $n_{\text{trans}}$ adjusts based on acceleration/deceleration direction
- $D_{\text{trans}}$ represents transition duration (max 2 gait cycles)

**Contact Sequence During Transition:**

Updates contact sequences backward (reverse pass) to maintain timing linearity:
```cpp
for i = GAIT_SEQUENCE_SIZE down to 0:
    v = velocity at timestep i (from MPC trajectory)
    Fr = calculate_froude(v, height)
    w1, w2 = get_transition_weights(Fr, time)
    
    for each leg:
        contact[i][leg] = (phase[i] + offset) < interpolated_duty_factor
        swing_time[i][leg] = (1 - phase[i]) * interpolated_period
```

**Early Contact Handling:**
- Detects foot contact in second half of planned swing phase
- Extends stance phase by setting `early_contact_possible` flag
- Forward pass applies detected contacts to sequence

**Advantages:**
- Biologically-inspired gait selection (matches animal locomotion)
- Smooth transitions prevent sudden jerky motions
- Adaptive to different speeds automatically

---

### 3. **AdaptiveGaitSequencer** (Dynamic Parameter Adaptation)

**Type:** Fully adaptive gait with real-time parameter optimization

**Key Characteristics:**
- Continuously adapts gait parameters (period, duty factor, phase offset) based on:
  - Measured/commanded velocity
  - Robot height
  - Froude number
  - Disturbance signals
  - Phase error correction
- Computes stride length from locomotion speed using empirical relationship
- Implements standing mode with optional foot repositioning via Raibert planner

**Parameter Adaptation Algorithm:**

**Step 1: Velocity Estimation**
```cpp
v = velocity_filter.update(reference_velocity).norm()
v_cmd = commanded_velocity.norm()
v = max(v, wz * 0.3)  // Account for rotational motion
v = max(v, v_cmd * min_v_cmd_factor)
v += disturbance_correction_factor * disturbance_signal
v = max(v, min_velocity)
```

**Step 2: Stride Length Calculation**
```cpp
Fr = v² / (g * h)
stride_length = (2.3 * Fr^0.3) * h  // Empirical relationship
stride_length = min(stride_length, max_stride_length)
```

**Step 3: Gait Parameter Updates**
```cpp
if v > zero_velocity_threshold:
    T_new = stride_length / v  // Dynamic period
    duty_factor[i] = (T_new - swing_time) / T_new  // Maintain swing time
else:
    T_new = correction_period  // Standing mode
    // Enable foot repositioning if needed
```

**Step 4: Phase Offset Adaptation**
- Switches between trot (`[0, 0.5, 0.5, 0]`) and walk (`[0, 0.5, 0.75, 0.25]`) based on Froude bounds
- Optional offset switching with configurable delay
- Prevents phase offset jitter

**Step 5: Phase Update with Error Correction**

Updates individual leg phases while correcting phase offset errors:

```cpp
for each leg i:
    // Standard phase advance
    phase[i] += dt / T
    
    // Scale stance/swing phases if duty factor changed
    if phase in stance phase:
        phase[i] *= (df_new / df_old)
    else:
        phase[i] = (phase - df_old) * (1-df_new)/(1-df_old) + df_new
    
    // Correct phase offset errors
    error = phase_offset[i] - current_phase_offset[i]
    max_correction = dt / (max_correction_cycles * T)
    phase[i] += clamp(error, -max_correction, max_correction)
```

**Standing Mode Behavior:**
- When velocity < threshold and all legs can move:
  - Uses Raibert foothold planning to generate desired foot positions
  - Checks distance between current and planned positions
  - Enables foot swing only if error > threshold or leg in swing phase
  - Maintains slow correction cycles (correction_period ≈ 0.4-1.0s)

**Disturbance Correction:**
```cpp
disturbance_factor = |v_actual - v_commanded|
v_adapted = v_measured + disturbance_correction * disturbance_factor
```

**Key Parameters:**
- `swing_time`: Fixed swing phase duration (typically 0.1-0.2s)
- `gait_change_froude`: Froude bounds for trot/walk switching
- `max_correction_cycles`: Number of cycles to correct phase offset (typically 2-5)
- `standing_foot_position_threshold`: Distance threshold for foot repositioning (typically 0.05m)
- `offset_delay`: Time delay before applying new phase offsets

---

## Shared Data Structures

### GaitSequence Structure

All sequencers populate this structure for each control step:

```cpp
struct GaitSequence {
    // Current state
    StateInterface::TimePoint time_stamp;
    Eigen::Vector3d target_position/velocity/orientation;
    double target_height, current_height;
    
    // Contact patterns (GAIT_SEQUENCE_SIZE steps ahead)
    array<array<bool, N_LEGS>, GAIT_SEQUENCE_SIZE> contact_sequence;
    array<array<double, N_LEGS>, GAIT_SEQUENCE_SIZE> swing_time_sequence;
    
    // Foot positions planned by Raibert planner
    array<array<Eigen::Vector3d, N_LEGS>, GAIT_SEQUENCE_SIZE> foot_position_sequence;
    
    // Reference trajectory from MPC
    array<Eigen::Vector3d, GAIT_SEQUENCE_SIZE> reference_trajectory_position;
    array<Eigen::Quaterniond, GAIT_SEQUENCE_SIZE> reference_trajectory_orientation;
    array<Eigen::Vector3d, GAIT_SEQUENCE_SIZE> reference_trajectory_velocity;
    array<Eigen::Vector3d, GAIT_SEQUENCE_SIZE> reference_trajectory_twist;
    
    // Mode: KEEP (standing) or MOVE (walking)
    enum Mode { KEEP = 0, MOVE = 1 } sequence_mode;
    
    array<double, N_LEGS> gait_swing_time;  // Max swing time per leg
};
```

### GaitSequencerInterface

All implementations inherit from this interface:

```cpp
class GaitSequencerInterface {
    virtual void GetGaitSequence(GaitSequence& gait_sequence) = 0;
    virtual void UpdateState(const StateInterface& quad_state) = 0;
    virtual void UpdateTarget(const Target& new_target) = 0;
    virtual void UpdateModel(const ModelInterface& quad_model) = 0;
    virtual void GetGaitState(interfaces::msg::GaitState& state) = 0;
    virtual GS_Type GetType() const = 0;
};
```

---

## Integration with MPC and Foothold Planning

### 1. MPC Integration

**MPC Trajectory Planner (`MPCTrajectoryPlanner`):**
- Called first in every control step: `plan_trajectory(gait_sequence, target)`
- Generates desired body trajectory (position, velocity, orientation, twist)
- Computes reference trajectory for the full prediction horizon
- Detects sequence mode (KEEP for standing, MOVE for walking)

**Flow:**
```
MPC planner → fills reference_trajectory_* 
              → gait sequencer reads these values
              → computes contact sequence aligned with MPC prediction
```

### 2. Raibert Footstep Planner Integration

**RaibertFootStepPlanner:**
- Computes foot target positions based on:
  - Current body state and commanded velocity
  - Stance phase duration (from gait sequencer)
  - Feedback control using position and velocity errors

**Algorithm:**
```cpp
p_foothold = p_shoulder + p_symmetry + p_centrifugal + p_correction

where:
  p_symmetry = average_velocity * stance_time / 2
  p_centrifugal = corrects for body rotation (angular velocity)
  p_correction = feedback term for trajectory tracking error
```

**Velocity Filtering:**
- Applies moving average filter to 2D velocity for smooth foothold planning
- Configurable filter size (typically 5-10 steps)
- Prevents foothold jitter from measurement noise

**Integration with Gait Sequencers:**
```cpp
// In each sequencer's GetGaitSequence():
foot_step_planner_.get_foot_position_sequence(gait_sequence, gait_);
// Fills gait_sequence.foot_position_sequence
```

---

## Early Contact Detection

**Mechanism (Optional, enabled per sequencer):**

When foot touches ground before planned contact:

1. **Detection Condition:**
   - Current phase > 50% of duty factor (in swing phase)
   - Actual contact detected (from foot sensors)
   - Planned contact not yet occurred

2. **Action:**
   - Mark leg for early contact
   - Extend stance phase until first planned contact
   - Maintains phase continuity

3. **Implementation in Gait Update:**
   ```cpp
   if (contact_detected && !planned_contact && phase > (duty_factor + 1.0) / 2.0) {
       early_contact_possible[leg] = true;
   }
   if (planned_contact) {
       early_contact_possible[leg] = false;
   } else if (early_contact_possible[leg]) {
       contact = true;  // Force contact in sequence
   }
   ```

**Benefits:**
- Handles uneven terrain better
- Reduces loss of balance during stumbles
- Maintains stability without requiring terrain knowledge

---

## Key Algorithms and Parameters

### Froude Number (BioInspired & Adaptive)

$$Fr = \frac{v^2}{g \cdot h}$$

- $v$: Forward velocity (m/s)
- $g$: Gravity (9.81 m/s²)
- $h$: Robot height (m)

Used to determine gait suitability and transition timing

### Stride Length Adaptation (AdaptiveGait)

$$stride\_length = 2.3 \cdot Fr^{0.3} \cdot h$$

Empirically-derived relationship matching animal locomotion principles

### Phase Continuity During Transitions

Ensures smooth contact sequence by:
1. Backward pass through prediction horizon
2. Computing interpolated parameters at each timestep
3. Forward pass applying early contact detection
4. Maintains linearity of swing times

### Contact Sequence Size

**GAIT_SEQUENCE_SIZE:** Typically 32-64 timesteps
- MPC prediction horizon length
- Control timestep dt ≈ 0.01-0.02s
- Covers ~0.5-1.0s of future contact prediction

---

## Supported Gaits

### Base Gait Database

```
STAND:              T=0.5s,   df=1.0,  offset=[0, 0, 0, 0]
STATIC_WALK:        T=1.25s,  df=0.8,  offset=[0, 0.5, 0.75, 0.25]
WALKING_TROT:       T=0.5s,   df=0.6,  offset=[0, 0.5, 0.5, 0]
TROT:               T=0.5s,   df=0.5,  offset=[0, 0.5, 0.5, 0]
FLYING_TROT:        T=0.4s,   df=0.4,  offset=[0, 0.5, 0.5, 0]
PACE:               T=0.35s,  df=0.5,  offset=[0, 0.5, 0, 0.5]
BOUND:              T=0.4s,   df=0.4,  offset=[0, 0, 0.5, 0.5]
ROTARY_GALLOP:      T=0.4s,   df=0.2,  offset=[0, 0.857, 0.357, 0.5]
TRAVERSE_GALLOP:    T=0.5s,   df=0.2,  offset=[0, 0.857, 0.357, 0.5]
PRONK:              T=0.5s,   df=0.5,  offset=[0, 0, 0, 0]
```

### BioInspired Gait Database

Same gaits but with Froude bounds:
```
STAND:              Fr∈[0, 0.0009]
STATIC_WALK:        Fr∈[0.0009, 0.0024]
WALKING_TROT:       Fr∈[0.0024, 0.1517]
TROT:               Fr∈[0.1517, 0.74]
FLYING_TROT:        Fr∈[0.74, 1.0]
```

---

## Configuration and Tuning

### SimpleGaitSequencer Parameters
- `raibert_filtersize`: 5-10
- `raibert_z_on_plane`: true/false (constrain height)
- `early_contact_detection`: true/false
- `fix_standing_position`: true/false

### BioInspiredGaitSequencer Parameters
- Same as Simple, plus Froude number automatic gait selection
- Transition smoothness controlled by `get_Dtrans()` (max 2 cycles)

### AdaptiveGaitSequencer Parameters
- `swing_time`: 0.1-0.2s
- `filter_size`: 10-20 (velocity filter)
- `zero_velocity_threshold`: 0.05-0.1 m/s
- `gait_change_froude`: [0.15, 0.5] (trot/walk switching bounds)
- `max_correction_cycles`: 2-5
- `standing_foot_position_threshold`: 0.05-0.1m
- `correction_period`: 0.4-1.0s
- `disturbance_correction`: 0.5-2.0
- `max_stride_length`: 0.3-0.5m

---

## Applying Adaptive Gait Sequencing to `pympc` (Recommended Snippets)

The following snippets show a minimal, Python-native way to bring the DFKI adaptive gait concepts into your current `FootholdReferenceGenerator` flow. They are designed to drop directly into `quadruped_pympc/helpers/foothold_reference_generator.py` with minimal disruption.

### 1. Froude Number + Gait Selection

```python
def compute_froude(self, base_lin_vel_H: np.ndarray) -> float:
    speed = float(np.linalg.norm(base_lin_vel_H))
    height = float(self.hip_height) if self.hip_height is not None else 0.28
    return (speed ** 2) / (self.gravity_constant * height)

def select_gait(self, froude: float) -> str:
    if froude < 0.0009:
        return "stand"
    if froude < 0.0024:
        return "static_walk"
    if froude < 0.1517:
        return "walking_trot"
    if froude < 0.74:
        return "trot"
    return "flying_trot"
```

### 2. Adaptive Stance Time (Duty Factor Update)

```python
def adaptive_stance_time(self, base_lin_vel_H: np.ndarray) -> float:
    froude = self.compute_froude(base_lin_vel_H)
    # Example mapping (tune as needed)
    if froude < 0.1:
        duty_factor = 0.7  # slower gait
    elif froude < 1.0:
        duty_factor = 0.5 - 0.2 * froude
    else:
        duty_factor = 0.3  # fast gait

    duty_factor = float(np.clip(duty_factor, 0.25, 0.85))
    return duty_factor * self.stance_time
```

### 3. Smooth Transition Between Gaits

```python
def blend_stance_time(self, old_stance: float, new_stance: float, elapsed: float, duration: float) -> float:
    alpha = float(np.clip(elapsed / max(duration, 1e-3), 0.0, 1.0))
    return (1.0 - alpha) * old_stance + alpha * new_stance
```

### 4. Phase Offset Correction (Disturbance Handling)

```python
def apply_phase_correction(self, phase_offsets: np.ndarray, phase_error: np.ndarray, dt: float, T: float) -> np.ndarray:
    max_correction = dt / (5.0 * max(T, 1e-3))  # 5 correction cycles
    correction = np.clip(phase_error, -max_correction, max_correction)
    return phase_offsets + correction
```

### 5. Integrate Into `compute_footholds_reference`

Replace the fixed-stance time usage with the adaptive value:

```python
# Existing (fixed stance)
# delta_ref_H = (self.stance_time / 2.0) * ref_base_lin_vel_H

# Adaptive stance
adaptive_stance = self.adaptive_stance_time(base_lin_vel_H)
delta_ref_H = (adaptive_stance / 2.0) * ref_base_lin_vel_H
```

Optionally blend the stance time when the gait changes:

```python
new_gait = self.select_gait(self.compute_froude(base_lin_vel_H))
if new_gait != self.current_gait:
    self.transition_start_time = time.time()
    self.prev_stance_time = self.stance_time
    self.current_gait = new_gait

elapsed = time.time() - self.transition_start_time
adaptive_stance = self.blend_stance_time(self.prev_stance_time, adaptive_stance, elapsed, duration=0.5)
```

---

## Applying AdaptiveGaitSequencer to `pympc` (No Bio Gait Selection)

This scenario mirrors the **AdaptiveGaitSequencer** behavior: it *continuously adapts* stride length, period, and duty factor using velocity and disturbance cues, **without** discrete Froude-based gait switching. It can live alongside the bio-inspired snippets above.

### 1. Velocity Filtering + Disturbance Compensation

```python
def adaptive_speed(self, base_lin_vel_H: np.ndarray, ref_base_lin_vel_H: np.ndarray) -> float:
    self.base_vel_hist.append(base_lin_vel_H)
    v_mvg = float(np.linalg.norm(np.mean(list(self.base_vel_hist), axis=0)))
    v_cmd = float(np.linalg.norm(ref_base_lin_vel_H))

    disturbance = abs(v_mvg - v_cmd)
    v_adapted = v_mvg + self.disturbance_correction * disturbance
    return max(v_adapted, self.min_velocity)
```

### 2. Stride Length + Period (AdaptiveGait Formula)

```python
def adaptive_period(self, v_adapted: float) -> float:
    height = float(self.hip_height) if self.hip_height is not None else 0.28
    froude = (v_adapted ** 2) / (self.gravity_constant * height)
    stride = 2.3 * (froude ** 0.3) * height
    stride = float(np.clip(stride, 0.0, self.max_stride_length))
    return max(stride / max(v_adapted, 1e-3), self.min_period)
```

### 3. Duty Factor From Fixed Swing Time

```python
def adaptive_duty_factor(self, period: float) -> float:
    duty_factor = (period - self.swing_time) / max(period, 1e-3)
    return float(np.clip(duty_factor, 0.25, 0.85))
```

### 4. Standing Mode With Foot Repositioning Trigger

```python
def should_reposition_feet(self, foot_error_norm: float, v_adapted: float) -> bool:
    return v_adapted < self.zero_velocity_threshold and foot_error_norm > self.standing_foot_position_threshold
```

### 5. Integrate Into `compute_footholds_reference`

```python
v_adapted = self.adaptive_speed(base_lin_vel_H, ref_base_lin_vel_H)
period = self.adaptive_period(v_adapted)
duty_factor = self.adaptive_duty_factor(period)
adaptive_stance = duty_factor * period

delta_ref_H = (adaptive_stance / 2.0) * ref_base_lin_vel_H
```

**Notes:**
- `swing_time`, `disturbance_correction`, `min_velocity`, `min_period`, `max_stride_length`, and `standing_foot_position_threshold` should be added as configurable fields (mirroring the DFKI adaptive sequencer).
- This approach avoids discrete gait switching. Instead, stride/period/duty adapt *continuously* from the measured/commanded velocity.

---

## Control Architecture Integration

```
┌─────────────────────────────────────┐
│   MIT Controller Main Loop          │
│   (Frequency: 200-500 Hz)           │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Update Robot State & Model         │
│  (From sensors/simulation)          │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  MPC Trajectory Planner             │
│  (Generates body reference traj)    │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Gait Sequencer                     │
│  ├─ Adaptive/Bio/Simple             │
│  ├─ Computes contact sequence       │
│  └─ Updates phase offsets           │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Raibert Footstep Planner           │
│  (Computes foot positions)          │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  WBC (Whole-Body Controller)        │
│  (Generates joint torques)          │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  Motor Drivers / Simulation         │
└─────────────────────────────────────┘
```

---

## Changes to `pympc`

This section documents the exact code changes added to enable **AdaptiveGaitSequencer-style** behavior in `pympc` without bio-inspired gait switching.

### 1. Configuration (`quadruped_pympc/config.py`)

Added adaptive gait parameters under `simulation_params`:

- `adaptive_gait_enabled` (bool): master toggle
- `adaptive_swing_time` (float)
- `adaptive_min_velocity` (float)
- `adaptive_min_period` (float)
- `adaptive_max_stride_length` (float)
- `adaptive_zero_velocity_threshold` (float)
- `adaptive_standing_foot_position_threshold` (float)
- `adaptive_disturbance_correction` (float)

**Added snippet (inside `simulation_params`, near gait/scene settings):**

```python
    # Adaptive gait sequencing (DFKI AdaptiveGait-style)
    'adaptive_gait_enabled':       False,
    'adaptive_swing_time':          0.15,
    'adaptive_min_velocity':        0.05,
    'adaptive_min_period':          0.2,
    'adaptive_max_stride_length':   0.4,
    'adaptive_zero_velocity_threshold': 0.05,
    'adaptive_standing_foot_position_threshold': 0.05,
    'adaptive_disturbance_correction': 1.0,
```

**Where each adaptive parameter is used:**

- `adaptive_gait_enabled`: gate for adaptive updates in `quadruped_pympc/interfaces/wb_interface.py` inside `WBInterface.update_state_and_reference(...)`.
- `adaptive_swing_time`: used in `quadruped_pympc/helpers/foothold_reference_generator.py` in `FootholdReferenceGenerator.adaptive_duty_factor(...)` (affects stance vs. swing split).
- `adaptive_min_velocity`: used in `FootholdReferenceGenerator.adaptive_speed(...)` to clamp very low speeds.
- `adaptive_min_period`: used in `FootholdReferenceGenerator.adaptive_period(...)` to avoid too-short gait periods.
- `adaptive_max_stride_length`: used in `FootholdReferenceGenerator.adaptive_period(...)` when clipping computed stride length.
- `adaptive_zero_velocity_threshold`: used in `FootholdReferenceGenerator.should_reposition_feet(...)` to define standing-mode behavior.
- `adaptive_standing_foot_position_threshold`: used in `FootholdReferenceGenerator.should_reposition_feet(...)` to decide if standing foot repositioning triggers.
- `adaptive_disturbance_correction`: used in `FootholdReferenceGenerator.adaptive_speed(...)` to boost speed estimate under disturbances.

### 2. Adaptive Gait Logic (`quadruped_pympc/helpers/foothold_reference_generator.py`)

Added AdaptiveGait-style helper methods and parameters:

- `adaptive_speed(...)`: moving-average velocity + disturbance correction
- `adaptive_period(...)`: stride length via Froude-based formula and adaptive period
- `adaptive_duty_factor(...)`: duty factor from fixed swing time
- `should_reposition_feet(...)`: standing-mode foot repositioning trigger

New adaptive params are pulled from `cfg.simulation_params` in the class constructor.

**Added snippet (inside `FootholdReferenceGenerator.__init__`):**

```python
        # Adaptive gait parameters (DFKI AdaptiveGait-style)
        self.swing_time = cfg.simulation_params.get('adaptive_swing_time', 0.15)
        self.min_velocity = cfg.simulation_params.get('adaptive_min_velocity', 0.05)
        self.min_period = cfg.simulation_params.get('adaptive_min_period', 0.2)
        self.max_stride_length = cfg.simulation_params.get('adaptive_max_stride_length', 0.4)
        self.zero_velocity_threshold = cfg.simulation_params.get('adaptive_zero_velocity_threshold', 0.05)
        self.standing_foot_position_threshold = cfg.simulation_params.get(
            'adaptive_standing_foot_position_threshold', 0.05
        )
        self.disturbance_correction = cfg.simulation_params.get('adaptive_disturbance_correction', 1.0)
```

**Added snippet (new helper methods below `__init__`):**

```python
    def adaptive_speed(self, base_lin_vel_h: np.ndarray, ref_base_lin_vel_h: np.ndarray) -> float:
        self.adaptive_vel_hist.append(base_lin_vel_h)
        velocity_avg = np.mean(list(self.adaptive_vel_hist), axis=0)
        measured_speed = float(np.linalg.norm(velocity_avg))
        commanded_speed = float(np.linalg.norm(ref_base_lin_vel_h))

        disturbance = abs(measured_speed - commanded_speed)
        adapted_speed = measured_speed + self.disturbance_correction * disturbance
        return max(adapted_speed, self.min_velocity)

    def adaptive_period(self, adapted_speed: float) -> float:
        height = float(self.hip_height) if self.hip_height is not None else float(cfg.hip_height)
        froude = (adapted_speed ** 2) / (self.gravity_constant * height)
        stride_length = 2.3 * (froude ** 0.3) * height
        stride_length = float(np.clip(stride_length, 0.0, self.max_stride_length))
        return max(stride_length / max(adapted_speed, 1e-3), self.min_period)

    def adaptive_duty_factor(self, period: float) -> float:
        duty_factor = (period - self.swing_time) / max(period, 1e-3)
        return float(np.clip(duty_factor, 0.25, 0.85))

    def should_reposition_feet(self, foot_error_norm: float, adapted_speed: float) -> bool:
        return adapted_speed < self.zero_velocity_threshold and foot_error_norm > self.standing_foot_position_threshold
```

### 3. Control-Loop Wiring (`quadruped_pympc/interfaces/wb_interface.py`)

In `WBInterface.update_state_and_reference(...)`:

- Compute adaptive speed/period/duty using `FootholdReferenceGenerator`
- Update `PeriodicGaitGenerator.step_freq` and `duty_factor` each cycle
- Update `FootholdReferenceGenerator.stance_time` from adaptive values
- Regenerate swing trajectory if swing period changes

**Added snippet (inside `update_state_and_reference`, before `self.pgg.run(...)`):**

```python
        if cfg.simulation_params.get('adaptive_gait_enabled', False):
            yaw = base_ori_euler_xyz[2]
            rotation_world_to_horizontal = np.array([np.cos(yaw), np.sin(yaw), -np.sin(yaw), np.cos(yaw)]).reshape((2, 2))
            base_lin_vel_h = rotation_world_to_horizontal @ base_lin_vel[0:2]
            ref_base_lin_vel_h = rotation_world_to_horizontal @ ref_base_lin_vel[0:2]

            adapted_speed = self.frg.adaptive_speed(base_lin_vel_h, ref_base_lin_vel_h)
            period = self.frg.adaptive_period(adapted_speed)
            duty_factor = self.frg.adaptive_duty_factor(period)

            self.pgg.step_freq = 1.0 / period
            self.pgg.duty_factor = duty_factor
            self.frg.stance_time = duty_factor * period

            swing_period = (1.0 - duty_factor) * period
            if abs(swing_period - self.last_swing_period) > 1e-4:
                self.stc.regenerate_swing_trajectory_generator(self.step_height, swing_period)
                self.last_swing_period = swing_period
```

This makes the contact sequence and foothold placement **continuously adaptive** to the measured/reference velocity.

### 4. Runtime Enablement

To activate adaptive gait sequencing, set:

```python
cfg.simulation_params['adaptive_gait_enabled'] = True
```

---

## Papers and References

### Published Work
1. **Benchmarking Different QP Formulations and Solvers for Dynamic Quadrupedal Walking**
   - Authors: Stark, Middelberg, Mronga, Vyas, Kirchner
   - Conference: ICRA 2025
   - ArXiv: http://arxiv.org/abs/2502.01329
   - Focus: MPC solver comparison, not gait sequencing

2. **Adaptive Model-Based Control of Quadrupeds via Online System Identification using Kalman Filter**
   - Status: Under review for IROS 2025
   - Focus: Model adaptation for robustness, integrates with adaptive MPC

### Key Research Areas Referenced in Code:
- **Bio-inspired gait selection**: Uses Froude number for automatic gait switching (biological principle)
- **Raibert's feedforward-feedback control**: For footstep planning
- **Phase-based contact representation**: Allows smooth transitions and phase error correction

---

## Comparison: Which Sequencer to Use?

| Aspect | Simple | BioInspired | Adaptive |
|--------|--------|------------|----------|
| **Complexity** | Low | Medium | High |
| **Computation** | Fast | Medium | Medium-Fast |
| **Speed Adaptation** | No | Automatic (Froude) | Fully Dynamic |
| **Gait Transitions** | None | Smooth blend | Continuous |
| **Parameter Tuning** | Simple gait params | Froude bounds | Many parameters |
| **Best For** | Fixed speed/gait | Variable speed | Terrain/disturbance handling |
| **Terrain Robustness** | Low | Medium | High |
| **Real-world Performance** | Good (flat ground) | Better | Best |

---

## Implementation Insights

### Why Three Sequencers?

1. **SimpleGaitSequencer**: Baseline for testing, high-speed execution, predictable timing
2. **BioGaitSequencer**: Bridges biological principles with engineering, automatic gait switching
3. **AdaptiveGaitSequencer**: Research platform for advanced adaptation, disturbance rejection

### Design Patterns Used

1. **Strategy Pattern**: `GaitSequencerInterface` with multiple implementations
2. **Template Method**: Base `Gait` class with specialized `BioInspiredGait`, `AdaptiveGait`
3. **Dependency Injection**: Sequencers receive `StateInterface`, `ModelInterface` as dependencies
4. **Moving Average Filtering**: Smooths sensor signals and velocity estimates

### Performance Considerations

- **Contact sequence precomputation**: Entire prediction horizon computed at once (vectorizable)
- **Phase-based representation**: Computation independent of absolute time, supports variable dt
- **Backward pass for transitions**: Ensures temporal consistency in BioInspired transitions
- **Filter-based velocity estimation**: Trades latency for noise robustness

---

## Future Enhancements Evident in Code

1. **TODO notes in BioGaitSequencer**: 
   - "maybe hard switch to new gait instead" - suggests exploration of direct switching vs. blending
   
2. **Disturbance correction in AdaptiveGait**:
   - `disturbance_correction_factor` suggests ongoing work on disturbance rejection
   
3. **Terrain awareness**:
   - Standing foot position threshold enables adaptive foot placement for terrain
   - Early contact detection handles unexpected obstacles

---

---

## Relationship Between Velocity, Duty Factor, and Period

### Non-Adaptive Case (adaptive_gait_enabled = False)

When adaptive gait is **disabled**, gait parameters are **fixed by configuration** and velocity affects **only foothold positioning**:

**Configuration (fixed at startup):**
```python
# From config.py
gait_params['trot'] = {
    'step_freq': 1.4,      # Hz → Period T = 1/1.4 ≈ 0.714 seconds
    'duty_factor': 0.65,   # Fraction of cycle in stance phase
    'type': GaitType.TROT
}
```

**Derived quantities (computed once, then constant):**
```
period (T) = 1 / step_freq = 1 / 1.4 ≈ 0.714 seconds
stance_time = (1 / step_freq) * duty_factor = 0.714 * 0.65 ≈ 0.464 seconds  # the time each leg is intended to stay in contact with the ground.
swing_time = (1 - duty_factor) / step_freq = 0.714 * 0.35 ≈ 0.25 seconds   #constant in the adaptive, as it is used to determine the duty factor.
```

**How velocity is used (foothold_reference_generator.py line 145):**
```python
delta_ref_H = (stance_time / 2.0) * ref_base_lin_vel_H
            = (0.464 / 2.0) * velocity
            = 0.232 * velocity  [meters]
```

**Velocity effects:**
- ✓ Affects: Foothold placement distance (footholds placed further ahead at higher velocities)
- ✗ Does NOT affect: `step_freq`, `period`, `duty_factor`, `stance_time`, `swing_time`, or contact timing

**Example foothold offsets (for trot):**
```
At 0.5 m/s:  offset = 0.116 m ahead
At 1.0 m/s:  offset = 0.232 m ahead
At 2.0 m/s:  offset = 0.464 m ahead
```

**Contact sequence generation (periodic_gait_generator.py):**
```python
phase[leg] += dt * step_freq  # step_freq = 1.4 Hz (CONSTANT)
contact[leg] = 1 if phase[leg] < duty_factor else 0
```
The contact timing is **independent of velocity**—legs lift and land at fixed times regardless of speed.

**Built-in velocity clipping (foothold_reference_generator.py line 145):**
```python
delta_ref_H = np.clip(delta_ref_H, -self.hip_height * 1.5, self.hip_height * 1.5)
```
With `hip_height = 0.28 m`:
```
max_offset = 0.42 m = 0.232 * v_max
→ v_max ≈ 1.81 m/s (MPC-derived limit)
```

This is why footholds become clipped (saturated) at high speeds in the non-adaptive case.

---

### Adaptive Case (adaptive_gait_enabled = True)

When adaptive gait is **enabled**, velocity drives **all three parameters continuously**:

**Velocity → Adapted Period (via Froude number):**
```python
froude = v² / (g * h)
stride_length = 2.3 * froude^0.3 * h
period = stride_length / v
```

**Period → Adapted Duty Factor:**
```python
duty_factor = (period - swing_time) / period
```

**Result:** Higher speeds produce shorter periods and lower duty factors (faster, lighter gait).

**Example adaptive behavior (for GO2, h=0.28 m):**
```
v = 0.5 m/s:  period ≈ 0.4s,  duty ≈ 0.63  (slow, supported)
v = 1.0 m/s:  period ≈ 0.35s, duty ≈ 0.57  (moderate trot)
v = 1.7 m/s:  period ≈ 0.32s, duty ≈ 0.53  (fast trot)
v = 2.2 m/s:  period ≈ 0.30s, duty ≈ 0.50  (flying trot)
```

**Key difference from non-adaptive:**
- Non-adaptive: Velocity only positions footholds, gait pattern stays the same
- Adaptive: Velocity reshapes the entire gait pattern (period shrinks, duty factor decreases)

---

### Practical Implications

**Why non-adaptive works:**
- Fixed gait pattern ensures predictable contact timing for MPC
- Velocity changes only affect "reach" not "rhythm"
- Simpler to tune and debug

**Why adaptive is needed:**
- Animals naturally use shorter, lighter steps at higher speeds
- Avoids foothold clipping (the 1.81 m/s MPC limit)
- Better terrain adaptation through continuous parameter tuning
- Disturbance rejection via velocity filtering

**Debugging tip:** If you see footholds "clipping" at certain velocities, you're hitting the non-adaptive case's velocity limit. Switching to adaptive mode or manually increasing `duty_factor` slightly will help.

## Conclusion

The DFKI quadruped controller provides a sophisticated gait sequencing framework that:

1. **Supports multiple sequencing strategies** (static, bio-inspired, fully adaptive)
2. **Seamlessly integrates with MPC** for coordinated body and contact planning
3. **Uses Raibert-based footstep planning** for stable foothold placement
4. **Implements early contact detection** for terrain robustness
5. **Provides smooth gait transitions** via interpolation and phase correction
6. **Enables real-time parameter adaptation** based on locomotion speed and disturbances

This architecture represents state-of-the-art in model-predictive control for quadrupedal locomotion, balancing computational efficiency with adaptive performance across diverse locomotion speeds and terrain conditions.
