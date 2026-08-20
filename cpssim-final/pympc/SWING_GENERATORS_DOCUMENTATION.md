# Swing Trajectory Generators - Comprehensive Documentation

## Overview

The swing generators are responsible for creating smooth, continuous 3D foot trajectories during the swing phase of locomotion. Two implementations are provided: **scipy-based (cubic splines)** and **explicit (Bézier curves)**. They generate desired positions, velocities, and accelerations for each foot during swing, which are then tracked by the swing trajectory controller using PD control with feedback linearization.

---

## Directory Structure

```
quadruped_pympc/helpers/swing_generators/
├── __init__.py
├── scipy_swing_trajectory_generator.py     # CubicSpline-based implementation
└── explicit_swing_trajectory_generator.py  # Bézier curve-based implementation
```

---

## 1. SCIPY SWING TRAJECTORY GENERATOR

### File: `scipy_swing_trajectory_generator.py`

This implementation uses **scipy's CubicSpline** interpolator to generate smooth foot trajectories.

#### Class: `SwingTrajectoryGenerator`

**Purpose:** Generate foot swing trajectories using cubic spline interpolation with 5 control points.

##### Constructor

```python
def __init__(self, step_height: float, swing_period: float) -> None
```

**Parameters:**
- `step_height` (float): Maximum height the foot raises during swing (default: 0.2 × hip_height ≈ 0.056m)
- `swing_period` (float): Total duration of swing phase (default: ~0.43s for trot gait)

**Instance Variables:**
- `self.step_height`: Copy of input step_height
- `self.swing_period`: Total swing duration
- `self.half_swing_period`: swing_period / 2
- `self.bezier_time_factor`: Computed factor for Bézier calculations (1 / half_swing_period)
- `self.reflex_next_steps_height_enhancement`: Flag to increase step height after early stance collision (reflex activation)
- `self.blind_locomotion`: Boolean - True if visual_foothold_adaptation == 'blind'
- `self.reflex_max_step_height`: Maximum step height when reflex is active (0.8 × hip_height ≈ 0.224m)
- `self._curve_x, _curve_y, _curve_z`: CubicSpline objects for x, y, z trajectories
- `self._curve_x_vel, _curve_y_vel, _curve_z_vel`: First derivative (velocity) splines
- `self._curve_x_acc, _curve_y_acc, _curve_z_acc`: Second derivative (acceleration) splines

---

#### Method: `createCurve()`

```python
def createCurve(self, x0, xf, early_stance_hitmoment=-1)
```

**Purpose:** Create CubicSpline curves for x, y, z trajectories with up to 5 control points.

**Parameters:**
- `x0` (np.array): Lift-off position [x, y, z] - where foot starts swing (world frame)
- `xf` (np.array): Touch-down position [x, y, z] - target landing position (world frame)
- `early_stance_hitmoment` (float): Time when early stance collision detected (-1 = no collision)

**Control Points Generated:**

**Case 1: Normal Swing (no early stance collision)**
```
p0 (lift-off):     x0 = [x0, y0, z0]
p1 (quarter-swing): x0 + [0, 0, step_height/1.5]
p2 (apex):         0.5*(x0 + xf) + [0, 0, step_height]        ← Highest point
p3 (3/4 swing):    xf + [0, 0, step_height/1.5]
p4 (touchdown):    xf = [xf, xf, zf]

Time points: [0, T/4, T/2, 3T/4, T]
```

**Case 2: Reflex Swing (early stance collision detected)**
```
p0 (collision pt):    early_stance_hitpoint
p1 (replan quarter):  x0 - 0.01*(xf-x0) + [0, 0, step_height/1.5]
p2 (replan apex):     0.5*(x0 + xf) + [0, 0, reflex_max_step_height]
p3 (replan 3/4):      0.2*x0 + 0.8*xf + [0, 0, reflex_max_step_height/1.5]
p4 (touchdown):       xf or xf + [0, 0, reflex_max_step_height/2.5]

Time points: [hitmoment, hitmoment+Δt/4, hitmoment+Δt/2, ...]
             where Δt = remaining_swing_time
```

**Spline Type:** Cubic spline with **clamped boundary conditions**
- Clamped BC: Assumes zero velocity at start and end of trajectory
- Results in smoother motion with natural acceleration profiles

**Generated Splines:**
```python
self._curve_x = CubicSpline(t, x, bc_type=["clamped", "clamped"])
self._curve_y = CubicSpline(t, y, bc_type=["clamped", "clamped"])
self._curve_z = CubicSpline(t, z, bc_type=["clamped", "clamped"])

# Derivative splines (for velocity and acceleration)
self._curve_x_vel = self._curve_x.derivative()
self._curve_y_vel = self._curve_y.derivative()
self._curve_z_vel = self._curve_z.derivative()

self._curve_x_acc = self._curve_x_vel.derivative()
self._curve_y_acc = self._curve_y_vel.derivative()
self._curve_z_acc = self._curve_z_vel.derivative()
```

---

#### Method: `compute_trajectory_references()`

```python
def compute_trajectory_references(self, swing_time, lift_off, touch_down, 
                                  early_stance_hitmoment=-1, 
                                  early_stance_hitpoint=None) 
    -> (np.array, np.array, np.array)
```

**Purpose:** Evaluate trajectory at current swing time to get desired foot state.

**Parameters:**
- `swing_time` (float): Current time within swing phase [0, swing_period]
- `lift_off` (np.array): Starting position of swing
- `touch_down` (np.array): Target landing position
- `early_stance_hitmoment` (float): Time of reflex collision (-1 if none)
- `early_stance_hitpoint` (np.array): Position where collision occurred

**Returns:**
```python
position = np.array([position_x, position_y, position_z])      # Desired foot position (m)
velocity = np.array([velocity_x, velocity_y, velocity_z])      # Desired foot velocity (m/s)
acceleration = np.array([acc_x, acc_y, acc_z])                 # Desired foot acceleration (m/s²)
```

**Computation Steps:**

1. **Create trajectory:** If reflex active, regenerate curve from collision point; else use normal curve
2. **Evaluate splines at swing_time:**
   ```python
   position_x = self._curve_x(swing_time)      # Spline evaluation
   velocity_x = self._curve_x_vel(swing_time)  # 1st derivative evaluation
   acceleration_x = self._curve_x_acc(swing_time)  # 2nd derivative evaluation
   ```
3. Same for y and z components
4. Return as reshaped numpy arrays

**Key Feature:** CubicSpline automatically ensures $C^2$ continuity (position, velocity, and acceleration are continuous) across all control points.

---

#### Visualization Methods

```python
def plot_trajectory_3d(self, curve_points: np.array) -> None
def plot_trajectory_references(self, tp, fp, vp, ap) -> None
```

Generate matplotlib plots showing:
- 3D trajectory path
- Position vs time (x, y, z components)
- Velocity vs time (x, y, z components)
- Acceleration vs time (x, y, z components)

---

### Characteristics

| Feature | Scipy Implementation |
|---------|---------------------|
| **Interpolation** | Cubic spline (piecewise polynomial) |
| **Control Points** | 5 points (lift-off, 2 intermediate, 2 near touchdown) |
| **Boundary Conditions** | Clamped (zero velocity at endpoints) |
| **Derivatives** | Exact (computed symbolically via scipy) |
| **Computational Cost** | ~0.1-0.2ms per leg per cycle |
| **Smoothness** | $C^2$ continuous (excellent) |
| **Reflection Support** | Yes - regenerates trajectory mid-swing |
| **Configuration** | Selected in `config.py` with `'swing_generator': 'scipy'` |

---

## 2. EXPLICIT SWING TRAJECTORY GENERATOR

### File: `explicit_swing_trajectory_generator.py`

This implementation uses **cubic Bézier curves** with explicit analytical derivatives.

#### Class: `SwingTrajectoryGenerator`

**Purpose:** Generate foot swing trajectories using Bézier curve formulation with analytical derivatives.

##### Constructor

```python
def __init__(self, step_height: float, swing_period: float) -> None
```

**Parameters:** Same as scipy version
- `step_height` (float)
- `swing_period` (float)

**Instance Variables:**
- `self.step_height`: Maximum swing height
- `self.swing_period`: Total duration
- `self.half_swing_period`: swing_period / 2
- `self.bezier_time_factor`: 1 / (swing_period / 2) - used for time normalization

**Note:** No `_curve_x` splines needed; derivatives computed analytically from Bézier basis functions.

---

#### Method: `compute_control_points()`

```python
def compute_control_points(self, swing_time, lift_off, touch_down) 
    -> (float, np.array, np.array, np.array, np.array)
```

**Purpose:** Determine 4 Bézier control points based on current swing phase.

**Parameters:**
- `swing_time` (float): Current time [0, swing_period]
- `lift_off` (np.array): Starting position
- `touch_down` (np.array): Landing position

**Returns:**
```python
(
    bezier_swing_time,  # swing_time normalized to [0, half_swing_period]
    cp1, cp2, cp3, cp4  # 4 Bézier control points
)
```

**Bézier Control Points:**

**First Half of Swing (swing_time ≤ half_swing_period):**
```
cp1 = lift_off                                    # P0
cp2 = lift_off                                    # P1
cp3 = [middle_x, middle_y, step_height]           # P2
cp4 = [middle_x, middle_y, step_height]           # P3
```

**Second Half of Swing (swing_time > half_swing_period):**
```
cp1 = [middle_x, middle_y, step_height]           # P0
cp2 = [touch_down_x, touch_down_y, step_height]   # P1
cp3 = touch_down                                  # P2
cp4 = touch_down                                  # P3
```

**Rationale:** Split swing into two phases:
- **Phase 1**: Accelerate from lift-off to midpoint (apex rising)
- **Phase 2**: Decelerate from midpoint to touchdown (descent)

---

#### Method: `compute_trajectory_references()`

```python
def compute_trajectory_references(self, swing_time, lift_off, touch_down,
                                  early_stance_hitmoment=-1,
                                  early_stance_hitpoint=None)
    -> (np.ndarray, np.ndarray, np.ndarray)
```

**Purpose:** Compute position, velocity, and acceleration using cubic Bézier formulas.

**Bézier Basis Functions:**

For normalized time $t \in [0, 1]$, cubic Bézier curve is:
$$\mathbf{B}(t) = (1-t)^3 \mathbf{P}_0 + 3t(1-t)^2 \mathbf{P}_1 + 3t^2(1-t) \mathbf{P}_2 + t^3 \mathbf{P}_3$$

**Position Computation:**
```python
s = self.bezier_time_factor * bezier_swing_time  # Normalized time [0, 1]

desired_foot_position = (
    (1 - s)**3 * cp1 +
    3 * s * (1 - s)**2 * cp2 +
    3 * s**2 * (1 - s) * cp3 +
    s**3 * cp4
)
```

**Velocity Computation:**

Derivative of Bézier curve:
$$\mathbf{B}'(t) = 3(1-t)^2(\mathbf{P}_1 - \mathbf{P}_0) + 6t(1-t)(\mathbf{P}_2 - \mathbf{P}_1) + 3t^2(\mathbf{P}_3 - \mathbf{P}_2)$$

```python
desired_foot_velocity = (
    3 * (1 - s)**2 * (cp2 - cp1) +
    6 * (1 - s) * s * (cp3 - cp2) +
    3 * s**2 * (cp4 - cp3)
)
```

**Acceleration Computation:**

Second derivative:
$$\mathbf{B}''(t) = 6(1-t)(\mathbf{P}_2 - 2\mathbf{P}_1 + \mathbf{P}_0) + 6t(\mathbf{P}_3 - 2\mathbf{P}_2 + \mathbf{P}_1)$$

```python
desired_foot_acceleration = (
    6 * (1 - s) * (cp3 - 2*cp2 + cp1) +
    6 * s * (cp4 - 2*cp3 + cp2)
)
```

**Returns:**
```python
(
    desired_foot_position.reshape((3,)),      # (m)
    desired_foot_velocity.reshape((3,)),      # (m/s)
    desired_foot_acceleration.reshape((3,))   # (m/s²)
)
```

---

#### Visualization Methods

```python
def plot_trajectory_3d(self, curve_points: np.ndarray) -> None
def plot_trajectory_references(self, tp, fp, vp, ap) -> None
```

Same as scipy version - generates matplotlib plots.

---

### Characteristics

| Feature | Explicit (Bézier) Implementation |
|---------|----------------------------------|
| **Interpolation** | Cubic Bézier curves |
| **Control Points** | 4 points (2 per half-swing) |
| **Boundary Conditions** | Not applicable (Bézier doesn't interpolate all points) |
| **Derivatives** | Analytical (exact closed-form) |
| **Computational Cost** | ~0.05-0.1ms per leg per cycle (faster) |
| **Smoothness** | $C^1$ continuous (smooth but not $C^2$) |
| **Reflection Support** | Limited - no mid-swing regeneration |
| **Configuration** | Selected in `config.py` with `'swing_generator': 'explicit'` |

---

## 3. SWING TRAJECTORY CONTROLLER

### File: `swing_trajectory_controller.py`

**Purpose:** Wrapper that selects between swing generators and computes swing control torques using feedback linearization.

#### Class: `SwingTrajectoryController`

**Key Responsibility:** Convert desired foot trajectories into joint torques via PD control.

##### Constructor

```python
def __init__(self, step_height, swing_period, position_gain_fb, 
             velocity_gain_fb, generator) -> None
```

**Parameters:**
- `step_height` (float): For swing generator initialization
- `swing_period` (float): For swing generator initialization
- `position_gain_fb` (float): PD position gain (typical: 500)
- `velocity_gain_fb` (float): PD velocity gain (typical: 10)
- `generator` (str): Choice of generator ('scipy' or 'explicit')

**Key Instance Variables:**
- `self.swing_generator`: Selected SwingTrajectoryGenerator instance
- `self.swing_time`: Array tracking swing time for each leg [0, 0, 0, 0]
- `self.use_feedback_linearization`: Flag for feedback linearization (typically True)
- `self.use_friction_compensation`: Flag to subtract passive friction forces (typically True)

---

##### Method: `regenerate_swing_trajectory_generator()`

```python
def regenerate_swing_trajectory_generator(self, step_height, swing_period) -> None
```

**Purpose:** Create new swing generator with updated parameters (called when gait changes).

**Used When:**
- Gait frequency is optimized (every 1-2 seconds)
- Reflex detects need for higher step height
- User changes gait parameters

**Example:** When trot frequency is optimized from 1.4 Hz to 1.6 Hz:
```
Old swing_period = 1/1.4 * (1 - 0.65) = 0.25s
New swing_period = 1/1.6 * (1 - 0.65) = 0.218s
Regenerate with new periods
```

---

##### Method: `compute_swing_control_cartesian_space()`

```python
def compute_swing_control_cartesian_space(self, leg_id, q_dot, J, J_dot, 
                                          lift_off, touch_down, foot_pos, 
                                          foot_vel, passive_force, h, 
                                          mass_matrix, 
                                          early_stance_hitmoments, 
                                          early_stance_hitpoints)
    -> (np.array, np.array, np.array)
```

**Purpose:** Compute swing leg control torques using end-effector space PD control.

**Parameters:**
- `leg_id` (int): Leg index (0-3 for FL, FR, RL, RR)
- `q_dot` (np.array): Joint velocities (3,)
- `J` (np.array): Foot Jacobian in world frame (3×3)
- `J_dot` (np.array): Time derivative of Jacobian (3×3)
- `lift_off` (np.array): Start position of swing
- `touch_down` (np.array): Target landing position
- `foot_pos` (np.array): Current foot position in world frame (3,)
- `foot_vel` (np.array): Current foot velocity in world frame (3,)
- `passive_force` (np.array): Passive friction/damping forces (3,)
- `h` (np.array): Joint bias forces (centrifugal, Coriolis, gravity) (3,)
- `mass_matrix` (np.array): Leg mass matrix (3×3)
- `early_stance_hitmoments` (float): Time of early stance collision
- `early_stance_hitpoints` (np.array): Position of collision

**Computation:**

```python
# Step 1: Get desired foot state from trajectory generator
des_foot_pos, des_foot_vel, des_foot_acc = self.swing_generator.compute_trajectory_references(
    self.swing_time[leg_id], lift_off, touch_down, 
    early_stance_hitmoments, early_stance_hitpoints
)

# Step 2: Compute position and velocity errors
err_pos = des_foot_pos - foot_pos        # (3,)
err_vel = des_foot_vel - foot_vel        # (3,)

# Step 3: Compute desired acceleration (with PD feedback)
acceleration = des_foot_acc + 
               position_gain_fb * err_pos + 
               velocity_gain_fb * err_vel    # (3,)

# Step 4: Compute torque without feedback linearization
tau_swing = J.T @ (position_gain_fb * err_pos + velocity_gain_fb * err_vel)

# Step 5: Apply feedback linearization
if self.use_feedback_linearization:
    tau_swing += mass_matrix @ pinv(J) @ (acceleration - J_dot @ q_dot) + h
```

**Feedback Linearization Details:**

The feedback linearization term ensures **acceleration tracking** by computing required joint acceleration:

$$\mathbf{a} = \ddot{\mathbf{x}}_{des} + K_p (\mathbf{x}_{des} - \mathbf{x}) + K_d (\dot{\mathbf{x}}_{des} - \dot{\mathbf{x}})$$

$$\boldsymbol{\tau} = \mathbf{J}^T \mathbf{F} + \mathbf{M} \mathbf{J}^{-T} (\mathbf{a} - \dot{\mathbf{J}} \dot{\mathbf{q}}) + \mathbf{h}$$

Where:
- $\mathbf{J}$: Jacobian (maps joint velocities to end-effector velocities)
- $\dot{\mathbf{J}} \dot{\mathbf{q}}$: End-effector acceleration from Jacobian time derivative
- $\mathbf{M}$: Leg mass matrix
- $\mathbf{h}$: Bias forces (gravity, Coriolis, centrifugal)

**Returns:**
```python
(
    tau_swing,       # Joint torques (3,) for swing phase tracking
    des_foot_pos,    # Desired foot position (3,) for logging
    des_foot_vel     # Desired foot velocity (3,) for logging
)
```

---

##### Method: `update_swing_time()`

```python
def update_swing_time(self, current_contact, legs_order, dt) -> None
```

**Purpose:** Increment swing time counter for each leg (called every control cycle).

**Logic:**
```python
for each leg:
    if leg is in swing (current_contact == 0):
        swing_time[leg] += dt        # Increment if swinging
    else:
        swing_time[leg] = 0          # Reset if in stance
```

**Called From:** `wb_interface.compute_stance_and_swing_torque()` every 2ms

**Example Flow:**
```
t=0ms:    All legs in contact → swing_time = [0, 0, 0, 0]
t=2ms:    Trot lifts FR, RL  → swing_time = [0, 2ms, 0, 2ms]
t=4ms:    FR, RL swinging    → swing_time = [0, 4ms, 0, 4ms]
...
t=250ms:  FR, RL touchdown   → swing_time = [0, 0, 0, 0]
```

---

##### Method: `check_apex_condition()`

```python
def check_apex_condition(self, current_contact, interval=0.02) -> int
```

**Purpose:** Detect when leg reaches swing apex (highest point).

**Logic:** Apex occurs at swing_period/2 ± interval

```python
if (swing_period/2 - 0.02) < swing_time < (swing_period/2 + 0.02):
    return 1 (apex detected)
else:
    return 0
```

**Uses:** Signal to optimize gait parameters (frequency, duty factor) at apex for stability

---

##### Method: `check_full_stance_condition()`

```python
def check_full_stance_condition(self, current_contact) -> int
```

**Purpose:** Detect when all four legs are in contact (full stance).

**Logic:**
```python
if all(current_contact == 1):
    return 1 (full stance)
else:
    return 0
```

**Used by:** Visual foothold adaptation, early stance detector reset

---

##### Method: `check_touch_down_condition()`

```python
def check_touch_down_condition(self, current_contact, previous_contact, 
                               contact_sequence, lookahead=3) -> int
```

**Purpose:** Detect stable touchdown of all feet (end of swing phase).

**Purpose:** Signal when to optimize step frequency.

**Rising Edge Detection:**
1. Detects transition from "at least one foot swinging" → "all feet in contact"
2. Waits for stability: Next 3 timesteps all remain in contact
3. Confirms next phase will have swing (lookahead timestep has foot lift)

**Returns:** 1 (trigger gait optimization), 0 (no action)

**Timing:** Called once per gait cycle at touchdown to reoptimize if needed

---

## 4. INTEGRATION WITH CODEBASE

### 4.1 Configuration Selection

**File:** `config.py` line 195

```python
simulation_params = {
    'swing_generator': 'scipy',  # Choose between 'scipy' or 'explicit'
    'swing_position_gain_fb': 500,    # PD position gain
    'swing_velocity_gain_fb': 10,     # PD velocity gain
    'step_height': 0.2 * hip_height,  # Default: ~0.056m for Go2
    'reflex_max_step_height': 0.8 * hip_height,  # Default: ~0.224m
    ...
}
```

**Runtime Impact:**
- Default scipy: Uses exact spline derivatives (best smoothness)
- Explicit: Faster computation, slightly less smooth derivatives

---

### 4.2 Whole-Body Interface Initialization

**File:** `wb_interface.py` lines 76-82

```python
class WBInterface:
    def __init__(self, ...):
        ...
        swing_generator = cfg.simulation_params['swing_generator']
        
        self.stc = SwingTrajectoryController(
            step_height=cfg.simulation_params['step_height'],
            swing_period=swing_period_from_gait,
            position_gain_fb=cfg.simulation_params['swing_position_gain_fb'],
            velocity_gain_fb=cfg.simulation_params['swing_velocity_gain_fb'],
            generator=swing_generator
        )
```

**When Called:** During robot initialization (once per simulation)

---

### 4.3 Control Loop Integration

**File:** `wb_interface.py` lines 303-430 (compute_stance_and_swing_torque method)

```
┌─────────────────────────────────────────────────────┐
│ Main Control Loop (100Hz MPC, ~500Hz swing control) │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ 1. Update Early Stance       │
        │    Detector (reflex)          │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │ 2. Compute Stance Torques     │
        │    (MPC GRF → Joint τ)        │
        └───────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────────────┐
        │ 3. Update Swing Times                 │
        │    (increment swing_time[leg_id])     │
        └───────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────┐
        │ 4. For each LEG in SWING phase:          │
        │                                          │
        │  a) Get desired trajectory from swing    │
        │     generator @ current swing_time       │
        │                                          │
        │  b) Compute PD errors:                   │
        │     err_pos = des_pos - actual_pos       │
        │     err_vel = des_vel - actual_vel       │
        │                                          │
        │  c) Compute swing torques with           │
        │     feedback linearization               │
        │                                          │
        │  d) Return [τ, des_pos, des_vel]        │
        └──────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────┐
        │ 5. Friction Compensation                 │
        │    (subtract passive forces if enabled)  │
        └──────────────────────────────────────────┘
                        │
                        ▼
        ┌──────────────────────────────────────────┐
        │ 6. Send Joint Torques to Actuators       │
        └──────────────────────────────────────────┘
```

---

### 4.4 Detailed Call Flow

#### Scenario: FL Leg in Swing Phase

```python
# In wb_interface.py compute_stance_and_swing_torque()

# Current state
swing_time[0] = 0.1s           # FL has been swinging for 100ms
lift_off = [0.1, 0.15, 0.0]   # Where FL took off
touchdown = [0.25, 0.15, 0.0] # Where FL should land

# Get trajectory reference
des_pos, des_vel, des_acc = stc.swing_generator.compute_trajectory_references(
    swing_time=0.1,            # Current swing time
    lift_off=lift_off,
    touch_down=touchdown,
    early_stance_hitmoment=-1  # No collision
)

# Trajectory generator returns:
# des_pos = [0.175, 0.15, 0.045]  (halfway through swing, raised height)
# des_vel = [0.75, 0, 0.15]        (horizontal motion, upward component)
# des_acc = [0, 0, -2.5]           (starting to decelerate in z)

# Compute errors
actual_pos = [0.172, 0.150, 0.040]  # Actual foot position from kinematics
actual_vel = [0.70, 0, 0.14]        # Actual foot velocity

err_pos = [0.003, 0, 0.005]
err_vel = [0.05, 0, 0.01]

# Compute acceleration with PD feedback
acc = des_acc + 500*err_pos + 10*err_vel
    = [0, 0, -2.5] + 500*[0.003, 0, 0.005] + 10*[0.05, 0, 0.01]
    = [1.5, 0, -2.5 + 2.5 + 0.1] = [1.5, 0, 0.1]

# Compute torques via feedback linearization
τ = J.T @ F_pd + M @ J^{-T} @ (a - J_dot @ q_dot) + h

# Result: 3-element torque vector for FL leg
τ_FL ≈ [1.2, -0.8, 0.5] Nm
```

---

### 4.5 Early Stance Reflex Integration

**File:** `early_stance_detector.py` lines 123-127

```python
# Reflex trigger when foot detects obstacle mid-swing
if early_stance_detected:
    # Store collision info
    self.hitpoints[leg_name] = collision_position
    self.hitmoments[leg_name] = swing_time_at_collision
    
    # On next swing, trajectory regenerates from collision point
    stc.swing_generator.reflex_next_steps_height_enhancement = True
```

**Next swing cycle:**
```python
# createCurve() called with:
early_stance_hitmoment = 0.15  # Collision at 150ms into swing
early_stance_hitpoint = [0.18, 0.15, 0.025]  # Collision position

# New trajectory:
# - Starts from collision point instead of normal lift-off
# - Uses higher step_height (reflex_max_step_height)
# - Completes swing from collision time → touchdown
# - Allows foot to clear obstacle
```

---

### 4.6 Gait Optimization Integration

**File:** `wb_interface.py` lines 327-333

```python
# Called when optimizer finds better step frequency
if optimize_swing == 1:
    self.pgg.step_freq = optimized_freq  # e.g., 1.4 → 1.6 Hz
    self.frg.stance_time = (1/freq) * duty_factor
    swing_period = (1 - duty_factor) / freq
    
    # Regenerate swing trajectory for new timing
    self.stc.regenerate_swing_trajectory_generator(
        step_height=step_height,
        swing_period=swing_period
    )
```

---

## 5. DATA FLOW DIAGRAM

```
┌──────────────────────────────────────────────────────────────┐
│                    SIMULATION START                          │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. config.py specifies swing_generator = 'scipy' or          │
│    'explicit'                                                │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. WBInterface.__init__() creates:                           │
│    - SwingTrajectoryController                              │
│    - Internally loads selected swing generator from config  │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. CONTROL LOOP (every 2ms)                                 │
│                                                              │
│    a) Update swing_time[leg_id] for each leg               │
│    b) For each leg in swing:                               │
│       - Call swing_generator.compute_trajectory_references()│
│       - Get: position, velocity, acceleration              │
│       - Compute PD errors                                  │
│       - Apply feedback linearization                       │
│       - Return joint torques                               │
│    c) Send τ to actuators                                  │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. OPTIONAL: Early Stance Detector Triggers Reflex          │
│                                                              │
│    - Foot hits obstacle mid-swing                          │
│    - regenerate_swing_trajectory_generator() called with   │
│      higher step_height                                    │
└──────────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. OPTIONAL: Gait Frequency Optimization                    │
│                                                              │
│    - At touchdown, if better frequency found              │
│    - regenerate_swing_trajectory_generator() called with   │
│      new swing_period                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. KEY PARAMETERS AND TUNING

### Swing Height Adjustment

```python
# Normal swing: ~1/6 of leg length above ground
step_height = 0.2 * hip_height       # ~56mm for Go2

# Emergency maneuver: clear obstacle
reflex_max_step_height = 0.8 * hip_height  # ~224mm for Go2
```

**When to adjust:**
- Increase for tall grass/debris
- Decrease for tight spaces
- Disable reflex if robot getting stuck

### PD Control Gains

```python
position_gain_fb = 500  # Strong position tracking
velocity_gain_fb = 10   # Moderate damping

# Impact on behavior:
# ↑ Kp → Faster tracking, more oscillation, higher torques
# ↓ Kp → Slower tracking, smoother, lower torques
# ↑ Kd → Damped, stable but sluggish
# ↓ Kd → Less damping, quicker but risk oscillation
```

**Typical tuning:**
- Kp = 500 for tracking accuracy
- Kd = 10-20 for smooth damping

### Swing Generator Selection

```python
# Scipy (default):
# ✓ Smoother C² continuous derivatives
# ✓ Better for high-speed gaits
# ✗ Slightly slower computation

# Explicit Bézier:
# ✓ Faster computation
# ✓ Lower computational cost
# ✗ C¹ continuous only (less smooth acceleration)
# ✗ No mid-swing reflex regeneration support

# Recommendation: Use scipy for robust locomotion
```

---

## 7. COMPUTATIONAL COST SUMMARY

| Component | Cost per Leg | Cost per Cycle (4 legs) |
|-----------|--------------|------------------------|
| Swing time update | <0.01ms | <0.04ms |
| Trajectory generation | 0.1-0.2ms | 0.4-0.8ms |
| PD control | 0.05ms | 0.2ms |
| Feedback linearization | 0.1ms | 0.4ms |
| **Total swing control** | **~0.25ms** | **~1.0ms** |

**Control cycle timing (total):**
- MPC: ~5-10ms @ 100Hz
- Swing control: ~1ms
- Other (state update, I/O): ~1ms
- **Total: ~8-12ms per cycle** (well within 50Hz update rate)

---

## 8. COMMON ISSUES AND DEBUGGING

### Issue 1: Foot Oscillates During Swing
**Cause:** PD gains too high (overshoot)
**Solution:** Reduce Kp to 300-400, increase Kd to 15-20

### Issue 2: Foot Doesn't Reach Target Position
**Cause:** Insufficient step height or weak PD gains
**Solution:** Increase step_height, increase Kp, check Jacobian accuracy

### Issue 3: Early Stance Detection Doesn't Trigger Reflex
**Cause:** Reflex disabled in config, threshold not met
**Solution:** Set `reflex_trigger_mode = 'tracking'` in config, lower detection threshold

### Issue 4: Smooth Acceleration Limits Exceeded
**Cause:** Bézier implementation (explicit) has discontinuous 2nd derivative
**Solution:** Switch to scipy generator (`'swing_generator': 'scipy'`)

### Issue 5: Slow Simulation Performance
**Cause:** Scipy spline computation is bottleneck
**Solution:** Use explicit Bézier generator, or run at lower control frequency

---

## 9. EXAMPLE: COMPLETE SWING CYCLE

**Setup:**
- Gait: Trot (swing_period = 0.25s)
- Step height: 0.06m
- From: [0.1, 0.15, 0.0]
- To: [0.25, 0.15, 0.0]

**Scipy Trajectory Progression:**

| Time | swing_time | Position | Velocity | Acceleration |
|------|-----------|----------|----------|--------------|
| 0.0s | 0.0s | [0.10, 0.15, 0.00] | [0.00, 0.00, 0.00] | [0.00, 0.00, 0.40] |
| 0.05s | 0.05s | [0.13, 0.15, 0.02] | [0.50, 0.00, 0.50] | [0.50, 0.00, 1.20] |
| 0.125s | 0.125s | [0.175, 0.15, 0.045] | [0.70, 0.00, 0.60] | [0.40, 0.00, 0.20] |
| 0.188s | 0.188s | [0.22, 0.15, 0.04] | [0.75, 0.00, 0.20] | [0.10, 0.00, -0.80] |
| 0.25s | 0.25s | [0.25, 0.15, 0.00] | [0.00, 0.00, 0.00] | [0.00, 0.00, -0.40] |

**Observations:**
- Position: Smooth parabolic trajectory to apex at T/2
- Velocity: Peaks at apex, returns to zero at touchdown
- Acceleration: Positive initially (lift), negative at end (landing)
- Natural landing force reduction due to zero acceleration

---

## 10. PERIODIC GAIT GENERATOR

The `PeriodicGaitGenerator` determines when each leg should be in contact with the ground (stance) or in swing phase. This generates the contact sequence used by the MPC and controls the timing of swing trajectories.

### File: `quadruped_pympc/helpers/periodic_gait_generator.py`

### Purpose
Generate periodic contact patterns (gaits) for quadruped locomotion with different gait types and configurable frequencies.

### Class: `PeriodicGaitGenerator`

#### Constructor
```python
def __init__(self, duty_factor, step_freq, gait_type: GaitType, horizon):
```

**Parameters:**
- `duty_factor` (float): Fraction of gait cycle in stance phase (0.0-1.0). Typical: 0.5-0.7
- `step_freq` (float): Stepping frequency in Hz (gait cadence). Typical: 1-3 Hz
- `gait_type` (GaitType): Type of gait pattern (TROT, PACE, BOUNDING, etc.)
- `horizon` (int): Planning horizon for contact sequence prediction

**Instance Variables:**
- `self._phase_signal`: Normalized phase for each leg (0.0-1.0)
- `self.phase_offset`: Phase delay for each leg defining gait pattern
- `self.duty_factor`: Stance duty factor
- `self.step_freq`: Step frequency

#### Gait Types & Phase Offsets

```
TROT (diagonal legs move together):
  phase_offset = [0.5, 1.0, 1.0, 0.5]   # [FL, FR, RL, RR]
  FL/RR swing → FR/RL swing → repeat
  Duty factor: ~0.5 (50% stance)

PACE (lateral legs move together):
  phase_offset = [0.8, 0.3, 0.8, 0.3]
  FL/RL swing → FR/RR swing → repeat

BOUNDING (front & rear pairs move together):
  phase_offset = [0.5, 0.5, 0.0, 0.0]
  FL/FR swing → RL/RR swing → repeat

CRAWL variants (one leg at a time):
  CIRCULARCRAWL: [0.0, 0.25, 0.75, 0.5]
  BFDIAGONALCRAWL: [0.0, 0.25, 0.5, 0.75]
  BACKDIAGONALCRAWL: [0.0, 0.5, 0.75, 0.25]
  FRONTDIAGONALCRAWL: [0.5, 1.0, 0.75, 1.25]
```

#### Key Methods

##### Method: `run(dt, new_step_freq)`

Advances gait phase by one time step and returns contact state.

```python
def run(self, dt, new_step_freq):
    """
    Args:
        dt: Time step (seconds)
        new_step_freq: Current step frequency (Hz)
    
    Returns:
        contact: (np.ndarray) [1,1,1,1] stance or [0,0,0,0] swing per leg
    """
```

**Algorithm:**
```
For each leg:
  phase[leg] += dt * step_freq
  phase[leg] = phase[leg] % 1.0  # Wrap to [0, 1)
  
  if phase[leg] < duty_factor:
    contact[leg] = 1  # Stance
  else:
    contact[leg] = 0  # Swing
```

**Example:**
```
duty_factor = 0.5, step_freq = 2 Hz
Initial phase_offset (trot) = [0.5, 1.0, 1.0, 0.5]

Time = 0.0s:
  phase = [0.5, 1.0, 1.0, 0.5] → contact = [0, 0, 0, 0] (all swing)

Time = 0.125s (dt=0.125, phase += 0.25):
  phase = [0.75, 0.25, 0.25, 0.75] → contact = [1, 0, 0, 1] (FL/RR stance)

Time = 0.25s (dt=0.125):
  phase = [0.0, 0.5, 0.5, 0.0] → contact = [1, 1, 1, 1] (all stance)

Time = 0.375s:
  phase = [0.25, 0.75, 0.75, 0.25] → contact = [0, 1, 1, 0] (FR/RL stance)

Time = 0.5s:
  phase = [0.5, 1.0, 1.0, 0.5] → contact = [0, 0, 0, 0] (repeat)
```

##### Method: `compute_contact_sequence(contact_sequence_dts, contact_sequence_lenghts)`

Predicts contact sequence over MPC horizon without modifying internal state.

```python
def compute_contact_sequence(self, contact_sequence_dts, contact_sequence_lenghts):
    """
    Args:
        contact_sequence_dts: Time steps for each horizon segment
        contact_sequence_lenghts: Horizon length for each segment
    
    Returns:
        contact_sequence: (4, horizon) array of contact predictions
    """
```

### Integration with Swing Trajectory

**Relationship to swing_period:**

```
swing_period = (1 - duty_factor) × (1 / step_freq)

Example (Trot):
  duty_factor = 0.6
  step_freq = 2 Hz
  swing_period = (1 - 0.6) × (1/2) = 0.4 × 0.5 = 0.2 seconds
```

**Control Flow:**
```
PeriodicGaitGenerator.run() 
  → Returns contact sequence [0,0,0,0] or [1,1,1,1] per leg
  → Swing phase = contact = 0
  → SwingTrajectoryGenerator uses swing_period for trajectory timing
  → Stance phase = contact = 1
  → FootholdReferenceGenerator plans next foothold
```

---

## 11. FOOTHOLD REFERENCE GENERATOR

Computes target footholds (touchdown positions) for the MPC using geometric heuristics and velocity compensation.

### File: `quadruped_pympc/helpers/foothold_reference_generator.py`

### Purpose
Generate reference footholds that stabilize the robot, account for velocity, and adapt to terrain.

### Class: `FootholdReferenceGenerator`

#### Constructor
```python
def __init__(self, stance_time, lift_off_positions, vel_moving_average_length=20, hip_height=None):
```

**Parameters:**
- `stance_time` (float): Duration of stance phase (seconds)
- `lift_off_positions` (LegsAttr): Initial feet positions at start
- `vel_moving_average_length` (int): Window for velocity averaging (default: 20)
- `hip_height` (float): Robot hip height for scaling

#### Key Methods

##### Method: `compute_footholds_reference(...)`

Computes reference footholds for the MPC based on current state and velocity.

**Signature:**
```python
def compute_footholds_reference(
    self, base_position, base_ori_euler_xyz, base_xy_lin_vel, 
    ref_base_xy_lin_vel, hips_position, com_height_nominal
) -> LegsAttr:
```

**Parameters:**
- `base_position` (np.ndarray): COM position in world frame [x, y, z]
- `base_ori_euler_xyz` (np.ndarray): Base orientation [roll, pitch, yaw]
- `base_xy_lin_vel` (np.ndarray): Current [vx, vy] in world frame
- `ref_base_xy_lin_vel` (np.ndarray): Desired [vx, vy] in world frame
- `hips_position` (LegsAttr): Hip positions for all legs
- `com_height_nominal` (float): Nominal COM height

**Algorithm:**

```
1. Transform to hip-centric frame:
   R_W2H = Rotation matrix (yaw only, horizontal plane)
   
2. Compute velocity compensation:
   delta_ref_H = (stance_time / 2.0) × ref_base_lin_vel_H
   • Footholds placed ahead of hips based on desired velocity
   • If vel = +0.5 m/s, stance_time = 0.6s:
     → place foot ~0.15m ahead
   
3. Compute error compensation:
   error_comp = sqrt(h/g) × (base_vel_moving_avg - ref_vel)
   • Accounts for velocity tracking error
   • Stabilizes against disturbances
   
4. Compute reference feet in hip-centric frame:
   ref_feet_H = hip_pos_H + default_stance_width + velocity_offset + error_offset
   
5. Transform back to world frame:
   ref_feet_W = R_W2H.T @ ref_feet_H + base_position
```

**Example Computation:**

```python
# Given:
base_position = [0, 0, 0.3]  # body at origin, 0.3m high
ref_base_lin_vel = [0.5, 0, 0]  # moving +x at 0.5 m/s
stance_time = 0.6  # seconds
hips_position.FL = [0.15, 0.1, 0.0]  # relative to body

# Step 1: velocity compensation
delta_ref = (0.6 / 2.0) * [0.5, 0] = [0.15, 0]

# Step 2: reference feet
ref_feet_H.FL = hip_pos_H + stance_width + velocity_offset
             = [0.15, 0.1] + [0, 0.05] + [0.15, 0]
             = [0.30, 0.15]

# Step 3: world frame
ref_feet_W.FL = R_W2H.T @ [0.30, 0.15] + [0, 0]
              = [0.30, 0.15]  (no yaw rotation)
```

**Key Features:**
- Clips velocity offset to ±1.5×hip_height to prevent unrealistic footholds
- Maintains stance width offsets for stability
- Updates moving average of velocity for smooth adaptation
- Accounts for COM offset for manual balance correction

##### Method: `update_lift_off_positions(...)`

Updates `lift_off_positions` when a foot transitions from stance to swing (loses contact).

```python
def update_lift_off_positions(
    self, previous_contact, current_contact, feet_pos, legs_order, 
    gait_type, base_position, base_ori_euler_xyz
):
```

**Logic:**
```
if previous_contact[leg] == 1 and current_contact[leg] == 0:
  # Foot just lifted off
  lift_off_positions[leg] = feet_pos[leg]  # Current foot position is lift-off point
```

##### Method: `update_touch_down_positions(...)`

Updates `touch_down_positions` when a foot transitions from swing to stance (makes contact).

```python
def update_touch_down_positions(
    self, previous_contact, current_contact, feet_pos, legs_order, 
    gait_type, base_position, base_ori_euler_xyz
):
```

### Integration with Swing Trajectory

**Data Flow:**
```
PeriodicGaitGenerator.run() → contact_sequence
         ↓
FootholdReferenceGenerator.compute_footholds_reference()
         ↓
MPC optimizer (uses footholds as constraints)
         ↓
nmpc_footholds (output)
         ↓
WBInterface.compute_swing_control_cartesian_space()
         ↓
SwingTrajectoryController receives:
  • lift_off = self.frg.lift_off_positions[leg]
  • touch_down = nmpc_footholds[leg]
```

---

## 12. TERRAIN ESTIMATOR

Estimates terrain geometry (roll, pitch, height) from foot positions to enable terrain-adaptive control.

### File: `quadruped_pympc/helpers/terrain_estimator.py`

### Purpose
Estimate terrain slope and elevation in real-time using kinematic measurements (foot positions).

### Class: `TerrainEstimator`

#### Constructor
```python
def __init__(self):
```

**Instance Variables:**
- `self.terrain_roll` (float): Estimated roll angle (rad)
- `self.terrain_pitch` (float): Estimated pitch angle (rad)
- `self.terrain_height` (float): Estimated average terrain height (m)
- `self.roll_activated` (bool): Enable roll estimation (default: False)
- `self.pitch_activated` (bool): Enable pitch estimation (default: True)

#### Key Methods

##### Method: `compute_terrain_estimation(...)`

Estimates terrain geometry from stance foot positions.

**Signature:**
```python
def compute_terrain_estimation(
    self, base_position, yaw, feet_pos, current_contact
) -> (float, float, float):
```

**Parameters:**
- `base_position` (np.ndarray): COM position [x, y, z]
- `yaw` (float): Robot heading (radians)
- `feet_pos` (dict): Foot positions {'FL': [x,y,z], 'FR': [...], ...}
- `current_contact` (np.ndarray): Contact state [1,0,1,0] per leg

**Algorithm:**

```
1. Transform feet to body-centric frame:
   R_W2H = Rotation about z-axis (yaw only)
   feet_H = R_W2H @ (feet_W - base_pos)
   
2. Compute foot differences:
   front_diff = feet_H[FL] - feet_H[FR]  # Front-left to front-right
   back_diff = feet_H[RL] - feet_H[RR]   # Back-left to back-right
   left_diff = feet_H[FL] - feet_H[RL]   # Front-left to back-left
   right_diff = feet_H[FR] - feet_H[RR]  # Front-right to back-right
   
3. Compute angles from height differences:
   pitch = 0.5 × (arctan(|left_diff_z| / |left_diff_x|) +
                  arctan(|right_diff_z| / |right_diff_x|))
   
   roll = 0.5 × (arctan(|front_diff_z| / |front_diff_y|) +
                 arctan(|back_diff_z| / |back_diff_y|))
   
4. Adjust signs based on foot positions:
   if (front_diff_z + back_diff_z) < 0: roll = -roll
   if (left_diff_z + right_diff_z) > 0: pitch = -pitch
   
5. Apply exponential smoothing (EMA):
   roll_smooth = 0.99 × roll_old + 0.01 × roll_new
   pitch_smooth = 0.99 × pitch_old + 0.01 × pitch_new
   height_smooth = 0.2 × height_old + 0.8 × height_new
```

**Example:**

```
Scenario: Robot on slope tilted +pitch (uphill to front)

feet_pos:
  FL: [0.1, 0.15, -0.18]   left front
  FR: [0.1, -0.15, -0.15]  right front (0.03m higher)
  RL: [-0.1, 0.15, -0.18]  left back
  RR: [-0.1, -0.15, -0.18] right back

left_diff = [0.2, 0, 0] → height diff ≈ 0 → pitch ≈ 0
right_diff = [0.2, 0, +0.03] → arctan(0.03/0.2) ≈ 8.5° → pitch ≈ +4.25°

Result: pitch = +4.25° (uphill)
```

**Smoothing Effect:**

```
Time = 0: terrain_pitch = 0 (initialized)
Step 1: meas_pitch = +5°
  result = 0.99 × 0 + 0.01 × 5 = 0.05°
  
Step 2: meas_pitch = +5°
  result = 0.99 × 0.05 + 0.01 × 5 = 0.1495°
  
... after 100 steps:
  result ≈ +5° (converges slowly)
```

The high smoothing factor (0.99) creates low-pass filtering to reject measurement noise.

### Usage in Control Loop

**Typical Call:**
```python
# In WBInterface or simulation loop
terrain_roll, terrain_pitch, terrain_height = self.terrain_estimator.compute_terrain_estimation(
    base_position=robot_position,
    yaw=robot_yaw,
    feet_pos=current_feet_positions,
    current_contact=contact_state
)
```

**Integration Gaps (Current Implementation):**
- Not currently used by swing trajectory generator
- Not integrated with foothold optimization
- Not used for swing height adaptation
- Estimated values not passed to MPC

**Potential Future Uses:**
1. Adapt swing height to terrain slope
2. Modify foothold computation based on slope
3. Adjust MPC cost functions for terrain
4. Enable terrain-aware gait selection

---

## 13. EARLY STANCE DETECTOR

Detects unexpected foot contacts during swing phase (obstacles) and triggers reactive reflex responses.

### File: `quadruped_pympc/helpers/early_stance_detector.py`

### Purpose
Detect mid-swing collisions with obstacles and regenerate swing trajectories to avoid or clear them.

### Class: `EarlyStanceDetector`

#### Constructor
```python
def __init__(self, feet_geom_id: LegsAttr = None, 
             legs_order: tuple = ('FL', 'FR', 'RL', 'RR')):
```

**Instance Variables:**
- `self.early_stance` (LegsAttr): Detection state per leg (bool)
- `self.hitpoints` (LegsAttr): Collision point in world frame per leg
- `self.hitmoments` (LegsAttr): Swing time of collision per leg
- `self.activated` (bool): Enable/disable reflex (config: `reflex_trigger_mode`)
- `self.trigger_mode` (str): Detection method ('tracking' or 'geom_contact')

**Thresholds:**
- `early_stance_time_threshold` (float): 0.07s - Ignore collision near swing ends
- `relative_tracking_error_threshold` (float): 0.3 - Tracking error ratio threshold
- `absolute_min_distance_error_threshold` (float): 0.1m - Minimum distance threshold

#### Key Methods

##### Method: `update_detection(...)`

Main detection logic called every 2ms during swing.

**Signature:**
```python
def update_detection(
    self, feet_pos, des_feet_pos, lift_off, touch_down, 
    swing_time, swing_period, current_contact, 
    previous_contact, mujoco_contact, stc
):
```

**Two Detection Modes:**

**Mode 1: TRACKING (Trajectory Error Based)**

```
Monitors tracking error between desired and actual foot position.

if not activated:
  # Reflex disabled
  early_stance[leg] = False
  
elif trigger_mode == 'tracking':
  for each swinging leg:
    
    # Skip if in stance or near swing end
    if current_contact[leg] == 1:
      continue
    if swing_time[leg] > swing_period - 0.07:  # Last 7% of swing
      continue
    
    # Compute tracking error
    local_disp = des_feet_pos[leg] - feet_pos[leg]  # Desired - actual
    swing_disp = touch_down[leg] - lift_off[leg]   # Planned swing
    
    # Collision detected if:
    error_ratio = ||local_disp|| / ||swing_disp||
    
    if error_ratio > 0.3 AND ||local_disp|| > 0.1m:
      # Large tracking error → obstacle detected
      hitpoint = feet_pos[leg]
      hitmoment = swing_time[leg]
      early_stance[leg] = True  # Collision flag
```

**Example:**

```
Scenario: Foot hits obstacle mid-swing

Planned swing: lift_off[FL] = [0, 0, 0]
              touch_down[FL] = [0.2, 0, 0]
              ||swing_disp|| = 0.2m

At swing_time = 0.1s:
  des_foot_pos[FL] = [0.1, 0, 0.05]  (halfway)
  feet_pos[FL] = [0.08, 0, 0.02]     (obstacle blocks vertical motion)
  local_disp = [0.02, 0, 0.03]
  ||local_disp|| = 0.036m
  
  error_ratio = 0.036 / 0.2 = 0.18 < 0.3 → NO collision (small error)

Different scenario (steep obstacle):
  feet_pos[FL] = [0.07, 0, 0]        (much lower)
  local_disp = [0.03, 0, 0.05]
  ||local_disp|| = 0.058m
  
  error_ratio = 0.058 / 0.2 = 0.29 < 0.3 → borderline
  but also 0.058m > 0.1m? No → NO collision
  
  feet_pos[FL] = [0, 0, 0]           (very low, big obstacle)
  local_disp = [0.1, 0, 0.05]
  ||local_disp|| = 0.111m
  
  error_ratio = 0.111 / 0.2 = 0.555 > 0.3 AND 0.111 > 0.1
  → COLLISION DETECTED ✓
```

**Mode 2: GEOM_CONTACT (Contact Geometry Based)**

```
Uses MuJoCo collision detection for more accurate contact.

if trigger_mode == 'geom_contact':
  for each swinging leg:
    contact_points = get_collision_points(feet_geom_id[leg])
    
    for contact_point in contact_points:
      # Skip if collision near start/end
      if swing_time[leg] < 0.07 or swing_time[leg] > period - 0.07:
        continue
      
      # Collision angle check (Bézier-based)
      if angle_between(swing_disp, contact_point - feet_pos[leg]) < 60°:
        # Contact on expected swing path
        hitpoint = contact_point
        hitmoment = swing_time[leg]
        early_stance[leg] = True
```

#### Reflex Integration

When collision detected, swing trajectory is regenerated:

```python
# In update_detection():
if early_stance[leg] == True:
  gait_cycles_after_reflex_height_enanchement = 0
  
# Next swing cycle:
stc.swing_generator.createCurve(
  lift_off=hitpoint,           # Regenerate from collision point
  touch_down=planned_touch_down,
  early_stance_hitmoment=hitmoment,
  early_stance_hitpoint=hitpoint
)

# Higher step height (reflex_max_step_height = 0.8 × hip_height)
# Clears obstacle by swinging higher
```

#### Height Enhancement Logic

```python
if use_reflex_next_steps_height_enhancement:
  if contact_transition_detected:  # Foot touches ground
    gait_cycles_after_reflex += 1
  
  if 0 <= gait_cycles_after_reflex < max_gait_cycles_height_enhancement (6):
    swing_generator.reflex_next_steps_height_enhancement = True
    # Swing 6 gait cycles with enhanced height
  else:
    swing_generator.reflex_next_steps_height_enhancement = False
    # Return to normal height
```

### Control Flow Integration

```
┌─────────────────────────────────────────┐
│ WBInterface (100 Hz)                    │
│ ├─ For each leg in swing:               │
│ │  ├─ Compute swing_trajectory (500 Hz) │
│ │  └─ Track actual foot position        │
│ └─ Update EarlyStanceDetector           │
└────────────────┬────────────────────────┘
                 │
         ┌───────▼────────┐
         │ EarlyStanceDetector
         ├─ Compare desired vs actual
         ├─ Detect collision (error > threshold)
         ├─ Store hitpoint, hitmoment
         └─ Set reflex flag
                 │
         ┌───────▼────────┐
         │ SwingTrajectoryController
         ├─ Next swing cycle:
         ├─ Pass hitpoint to generator
         └─ Regenerate trajectory
                 │
         ┌───────▼────────┐
         │ SwingGenerator
         ├─ createCurve() from hitpoint
         ├─ Higher step_height (0.8×h)
         └─ New smooth trajectory
                 │
         ┌───────▼────────┐
         │ 6 gait cycles with enhancement
         └─ Return to normal height
```

### Configuration

From `config.py`:
```python
cfg.simulation_params['reflex_trigger_mode'] = 'tracking'  # or 'geom_contact' or False
cfg.simulation_params['reflex_next_steps_height_enhancement'] = True
```

### Current Limitations

1. **Reactive-only:** Detects collision after contact (no prediction)
2. **Single-point regeneration:** Regenerates from first collision point only
3. **No terrain adaptation:** Doesn't incorporate terrain slope
4. **Limited horizon:** Only reacts to immediate swing phase, not multi-step planning
5. **Thresholds hand-tuned:** Not adaptive to terrain roughness or speed

### Potential Improvements

1. Predictive obstacle detection (use terrain estimator)
2. Multi-point reflex regeneration
3. Learning-based threshold adaptation
4. Integration with MPC for terrain-aware planning

---

## 14. SYSTEM ARCHITECTURE SUMMARY

### Complete Swing Trajectory Generation Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CONTROL ARCHITECTURE                              │
└──────────────────────────────────────────────────────────────────────┘

MPC OPTIMIZATION (100 Hz)
├─ Receives: current state, terrain estimate
├─ Optimizes: footholds, GRF, trajectory
└─ Outputs: nmpc_footholds, nmpc_GRF

         ▼

FOOTHOLD REFERENCE GENERATOR
├─ Receives: nmpc_footholds, velocity, terrain
├─ Computes: reference footholds with velocity compensation
├─ Updates: lift_off positions when feet lose contact
└─ Outputs: touch_down position per leg

         ▼

PERIODIC GAIT GENERATOR
├─ Receives: duty_factor, step_freq, gait_type
├─ Generates: contact sequence
├─ Determines: swing_period = (1 - duty_factor) / step_freq
└─ Outputs: contact[leg] = 1 (stance) or 0 (swing)

         ▼

WB INTERFACE (100 Hz main loop, 500 Hz swing control)
├─ Receives: MPC outputs, contact state
├─ For each leg in SWING:
│  ├─ Get: lift_off = current_foot_position
│  ├─ Get: touch_down = nmpc_footholds
│  └─ Call: SwingTrajectoryController
│
├─ For each leg in STANCE:
│  └─ Call: FootholdReferenceGenerator to update lift_off
│
└─ Outputs: τ (joint torques)

         ▼

SWING TRAJECTORY CONTROLLER
├─ Receives: lift_off, touch_down, swing_time[leg]
├─ Calls: swing_generator.compute_trajectory_references()
├─ Returns: [desired_pos, desired_vel, desired_acc]
├─ Computes: PD errors + feedback linearization
└─ Outputs: τ_swing (joint torques)

         ▼

SWING GENERATOR (scipy or explicit)
├─ Receives: lift_off, touch_down, swing_period, swing_time
├─ Scipy: Creates 5-point CubicSpline with derivatives
├─ Explicit: Evaluates cubic Bézier basis functions
├─ Computes: smooth trajectory with position, velocity, acceleration
└─ Outputs: [pos(t), vel(t), acc(t)]

         ▼ (if collision detected)

EARLY STANCE DETECTOR (Reflex)
├─ Monitors: tracking error (desired - actual)
├─ Detects: collision if error > threshold
├─ Triggers: trajectory regeneration from hitpoint
├─ Increases: step_height for 6 gait cycles
└─ Re-enters: swing generator with new parameters

         ▼

TERRAIN ESTIMATOR (slow update)
├─ Estimates: roll, pitch, height from foot positions
├─ Uses: exponential smoothing (0.99/0.01 ratio)
├─ Currently: Not actively used in swing control
└─ Future: Could adapt swing height to terrain slope
```

### Component Interactions Table

```
┌──────────────────────────────┬────────────┬───────────────────────┐
│ Component                    │ Frequency  │ Key Outputs to Swing  │
├──────────────────────────────┼────────────┼───────────────────────┤
│ PeriodicGaitGenerator        │ 100 Hz MPC │ contact sequence      │
│                              │            │ swing_period formula  │
├──────────────────────────────┼────────────┼───────────────────────┤
│ FootholdReferenceGenerator   │ 100 Hz MPC │ lift_off_positions    │
│                              │ On stance  │ touch_down tracking   │
├──────────────────────────────┼────────────┼───────────────────────┤
│ TerrainEstimator            │ 100 Hz MPC │ roll, pitch, height   │
│                              │ (unused)   │ (not currently used)  │
├──────────────────────────────┼────────────┼───────────────────────┤
│ EarlyStanceDetector         │ 2 ms       │ hitpoint, hitmoment   │
│                              │ (on swing) │ reflex_height_flag    │
├──────────────────────────────┼────────────┼───────────────────────┤
│ SwingGenerator              │ 2 ms       │ [pos, vel, acc](t)    │
│ (scipy/explicit)            │ (on swing) │ at swing_time         │
└──────────────────────────────┴────────────┴───────────────────────┘
```

### Data Flow Diagram

```
lift_off ─────┐
              ├──► SwingGenerator ──► [pos, vel, acc]
touch_down ───┤                            ↓
              └── swing_time ────► SwingTrajectoryController
                                        ↓
                                   PD Control
                                        ↓
                                   Feedback Linearization
                                        ↓
                                    Joint Torques (τ)

During swing with reflex:
  
  feet_pos ──┐
             ├──► EarlyStanceDetector ──► hitpoint
  des_pos ───┤                            hitmoment
             └── swing_time ────► collision detected?
                                        ↓ YES
                                   SwingGenerator
                                   (regenerate from hitpoint)
                                        ↓
                                   New trajectory with
                                   higher step_height
```

---

## 15. REFERENCES

**Key Papers:**
- Bezier curves: *Bézier Curve Construction* in computer graphics literature
- CubicSpline: scipy.interpolate.CubicSpline documentation
- Quadruped control: *Recent work on dynamic quadruped locomotion*

**Code References:**
- scipy.interpolate.CubicSpline: Creates smooth interpolating splines
- Feedback linearization: Computed via Jacobian inverse and mass matrix

---

**Document Version:** 2.0  
**Last Updated:** February 1, 2026  
**Author:** Comprehensive Codebase Analysis with Supporting Systems  
**Sections Added:** Periodic Gait Generator, Foothold Reference Generator, Terrain Estimator, Early Stance Detector, System Architecture Summary


**Code References:**
- scipy.interpolate.CubicSpline: Creates smooth interpolating splines
- Feedback linearization: Computed via Jacobian inverse and mass matrix

---

**Document Version:** 1.0  
**Last Updated:** February 1, 2026  
**Author:** Comprehensive Codebase Analysis  
