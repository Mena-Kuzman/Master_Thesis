# SRBD Interface Outputs: Where They Are Used

## Overview

The SRBD Interface produces 7 outputs that are consumed by multiple downstream components. This document traces each output to its usage points.

```
┌─────────────────────────────────────────┐
│ SRBD Controller Interface               │
│ compute_control() returns 7 outputs     │
└────────┬────────────────────────────────┘
         │
         ├─→ nmpc_GRFs (4 legs × 3D forces)
         ├─→ nmpc_footholds (4 legs × 3D positions)
         ├─→ nmpc_joints_pos (4 legs × 3 angles, kinodynamic only)
         ├─→ nmpc_joints_vel (4 legs × 3 velocities, kinodynamic only)
         ├─→ nmpc_joints_acc (4 legs × 3 accelerations, kinodynamic only)
         ├─→ best_sample_freq (float, optimized step frequency)
         └─→ nmpc_predicted_state (np.ndarray, trajectory prediction)
```

---

## Part 1: nmpc_GRFs Usage

### Output Structure
```python
nmpc_GRFs = LegsAttr(
    FL=np.array([Fx, Fy, Fz]),
    FR=np.array([Fx, Fy, Fz]),
    RL=np.array([Fx, Fy, Fz]),
    RR=np.array([Fx, Fy, Fz]),
)
```

### Where it's Used

#### 1.1 Stance Torque Computation (PRIMARY USE)

**File:** `wb_interface.py`

**Function:** `compute_stance_and_swing_torque()`

**Lines:** 366-369

**Purpose:** Convert MPC-optimized ground reaction forces into joint torques via inverse dynamics

```python
def compute_stance_and_swing_torque(..., nmpc_GRFs: LegsAttr, ...):
    # Compute Stance Torque ---------------------------------------------------------------------------
    tau.FL = -np.matmul(feet_jac.FL[:, legs_qvel_idx.FL].T, nmpc_GRFs.FL)
    tau.FR = -np.matmul(feet_jac.FR[:, legs_qvel_idx.FR].T, nmpc_GRFs.FR)
    tau.RL = -np.matmul(feet_jac.RL[:, legs_qvel_idx.RL].T, nmpc_GRFs.RL)
    tau.RR = -np.matmul(feet_jac.RR[:, legs_qvel_idx.RR].T, nmpc_GRFs.RR)
```

**Formula:** `τ = -J^T · F_mpc`

Where:
- `J` = Foot Jacobian (3×3 matrix mapping joint torques to foot forces)
- `F_mpc` = MPC-computed ground reaction force (3D)
- Negative sign because we're computing reaction torque

**Data Flow:**
```
nmpc_GRFs (12D: 4 legs × 3D)
    ↓
Extract per-leg forces: nmpc_GRFs.FL, .FR, .RL, .RR
    ↓
Multiply by Jacobian transpose: J^T @ F
    ↓
Store in joint torques: τ[leg] = -J^T @ F
    ↓
Send to actuators/simulator
```

**Example Calculation:**
```
nmpc_GRFs.FL = [20 N, 0 N, 150 N]  (3D force in world frame)
feet_jac.FL = 3×3 matrix
τ.FL = -feet_jac.FL^T @ [20, 0, 150]
     = -[[...], [...], [...]] @ [20, 0, 150]
     ≈ [-2.5, -3.2, 1.8] Nm  (3D torque at hip)
```

**Why negative sign?**
- MPC computes desired foot forces (what floor exerts on foot)
- We need reaction torques at hip (inverse direction)
- Negative sign accounts for Newton's third law

---

#### 1.2 Display/Observation (SECONDARY USE)

**Files:**
- `quadruped_pympc_wrapper.py` (line 39-40)
- `quadruped_pympc_wrapper.py` line 9 (default observation)

**Purpose:** Include in environment observations for learning-based algorithms

```python
# In wrapper initialization
self.nmpc_GRFs = LegsAttr(FL=np.zeros(3), FR=np.zeros(3), RL=np.zeros(3), RR=np.zeros(3))

# Default observations include
_DEFAULT_OBS = ("ref_base_height", "ref_base_angles", "nmpc_GRFs", "nmpc_footholds", ...)
```

**Usage:** Provides MPC force output to reinforcement learning agents or tracking algorithms

---

## Part 2: nmpc_footholds Usage

### Output Structure
```python
nmpc_footholds = LegsAttr(
    FL=np.array([x, y, z]),  # World frame position
    FR=np.array([x, y, z]),
    RL=np.array([x, y, z]),
    RR=np.array([x, y, z]),
)
```

### Where it's Used

#### 2.1 Swing Trajectory Target (PRIMARY USE - SWING PHASE)

**File:** `wb_interface.py`

**Function:** `compute_stance_and_swing_torque()`

**Lines:** 389

**Purpose:** Set the desired touchdown position for swing phase feet

```python
for leg_id, leg_name in enumerate(self.legs_order):
    if self.current_contact[leg_id] == 0:  # SWING PHASE
        tau[leg_name], des_foot_pos[leg_name], des_foot_vel[leg_name] = \
            self.stc.compute_swing_control_cartesian_space(
                leg_id=leg_id,
                lift_off=self.frg.lift_off_positions[leg_name],
                touch_down=nmpc_footholds[leg_name],  # ← MPC foothold used here
                foot_pos=feet_pos[leg_name],
                foot_vel=feet_vel[leg_name],
                ...
            )
```

**Data Flow:**
```
nmpc_footholds[leg] = desired touchdown position
    ↓
Pass to SwingTrajectoryController
    ↓
Generate trajectory from (lift_off) to (touch_down)
    ↓
Track trajectory with PD control
    ↓
Compute swing torques
```

**What happens inside swing controller:**
1. Receives touchdown target from MPC
2. Generates smooth trajectory from current lift-off to target
3. Tracks trajectory with feedback control
4. Outputs joint torques to follow trajectory

---

#### 2.2 Stance Position Maintenance (PRIMARY USE - STANCE PHASE)

**File:** `wb_interface.py`

**Lines:** 400

**Purpose:** Maintain foot position during stance phase

```python
else:  # STANCE PHASE
    des_foot_pos[leg_name] = nmpc_footholds[leg_name]
    des_foot_vel[leg_name] = des_foot_vel[leg_name] * 0.0  # No movement
```

**Meaning:**
- During stance, foot stays at the MPC-computed foothold position
- No velocity is commanded (foot should be stationary)
- Position tracking maintains contact force

---

#### 2.3 Early Stance Detector Reference (SECONDARY USE)

**File:** `wb_interface.py`

**Lines:** 359

**Purpose:** Provide touchdown reference for collision detection and trajectory regeneration

```python
self.esd.update_detection(
    feet_pos, 
    self.last_des_foot_pos, 
    lift_off=self.frg.lift_off_positions, 
    touch_down=nmpc_footholds,  # ← Used as reference for collision checks
    swing_time=self.stc.swing_time,
    swing_period=self.stc.swing_period,
    current_contact=self.current_contact,
    previous_contact=self.previous_contact,
    mujoco_contact=mujoco_contact,
    stc=self.stc
)
```

**What happens:**
1. Early stance detector receives MPC touchdown target
2. Compares with actual foot position during swing
3. If actual contact is far from target trajectory → reflex triggered
4. Regenerates trajectory from impact point toward MPC target

**Example:**
```
MPC touchdown target: [0.3, 0.1, -0.28] m
Actual impact point: [0.32, 0.09, -0.26] m (hit obstacle)
Distance: 0.028 m (>0.1 m threshold)
→ Reflex trajectory regenerated from [0.32, 0.09, -0.26] toward target
```

---

#### 2.4 Observation/Logging (TERTIARY USE)

**Files:**
- `quadruped_pympc_wrapper.py` (lines 40, 137, 183)
- Default observations include `nmpc_footholds`

**Purpose:** Track what footholds MPC is commanding

---

## Part 3: nmpc_joints_pos Usage

### Output Structure (Kinodynamic Only)
```python
nmpc_joints_pos = LegsAttr(
    FL=np.array([θ1, θ2, θ3]),  # 3 joint angles in radians
    FR=np.array([θ1, θ2, θ3]),
    RL=np.array([θ1, θ2, θ3]),
    RR=np.array([θ1, θ2, θ3]),
)
```

### Where it's Used

#### 3.1 Joint Position Reference (PRIMARY USE - KINODYNAMIC ONLY)

**File:** `wb_interface.py`

**Lines:** 439

**Purpose:** Direct joint angle commands for kinodynamic MPC models

```python
if cfg.mpc_params['type'] != 'kinodynamic':
    # Compute joint positions from foot positions via inverse kinematics
    des_joints_pos = ... # IK computation
else:
    # KINODYNAMIC: Use MPC-computed joint positions directly
    des_joints_pos = nmpc_joints_pos  # ← Direct use
```

**Data Flow (Kinodynamic):**
```
nmpc_joints_pos = [θ1, θ2, θ3, ...] from MPC
    ↓
des_joints_pos = nmpc_joints_pos  (direct assignment)
    ↓
PD control tracking: τ = Kp(des_pos - actual_pos) + Kd(des_vel - actual_vel)
    ↓
Joint torques computed
```

**Data Flow (Non-Kinodynamic):**
```
nmpc_footholds = [x, y, z] foot position
    ↓
Inverse Kinematics: θ = IK(foot_pos)
    ↓
des_joints_pos = θ
    ↓
PD control tracking
    ↓
Joint torques
```

**Why the difference?**
- **Kinodynamic MPC:** Optimizes at joint level, outputs joint angles directly
- **Centroidal MPC:** Optimizes at foot level, needs IK to convert to joints

---

#### 3.2 Error Computation (SUPPORTING USE)

**File:** `swing_trajectory_controller.py`

**Lines:** 93-98

**Purpose:** Compute joint-level tracking errors in feedback linearization

```python
def compute_momentum_rate_nondimensional(
    self, nmpc_joints_pos, nmpc_joints_vel, nmpc_joints_acc, qpos, qvel, 
    legs_mass_matrix, legs_qfrc_bias, legs_qfrc_passive
):
    error_position = nmpc_joints_pos - qpos      # Joint position error
    error_velocity = nmpc_joints_vel - qvel      # Joint velocity error
    acceleration = nmpc_joints_acc               # Desired acceleration
```

---

## Part 4: nmpc_joints_vel Usage

### Output Structure (Kinodynamic Only)
```python
nmpc_joints_vel = LegsAttr(
    FL=np.array([ω1, ω2, ω3]),  # 3 joint velocities in rad/s
    FR=np.array([ω1, ω2, ω3]),
    RL=np.array([ω1, ω2, ω3]),
    RR=np.array([ω1, ω2, ω3]),
)
```

### Where it's Used

#### 4.1 Joint Velocity Reference (PRIMARY USE - KINODYNAMIC ONLY)

**File:** `wb_interface.py`

**Lines:** 440 (INCORRECT IN CODE - should be des_joints_vel)

**Note:** There's a bug in the code - line 440 assigns to `des_joints_pos` instead of `des_joints_vel`:
```python
# Line 439-440 (CURRENT - with bug):
des_joints_pos = nmpc_joints_pos
des_joints_pos = nmpc_joints_vel  # ← SHOULD BE des_joints_vel
```

**Intended Purpose:** Provide desired joint velocities for PD control

```python
# Should be:
des_joints_pos = nmpc_joints_pos
des_joints_vel = nmpc_joints_vel
```

#### 4.2 Velocity Error Computation (SUPPORTING USE)

**File:** `swing_trajectory_controller.py`

**Lines:** 98

**Purpose:** Compute velocity-level tracking errors

```python
error_velocity = nmpc_joints_vel - qvel
```

Used in feedback linearization: `τ = Kp·e_pos + Kd·e_vel + ...`

---

## Part 5: nmpc_joints_acc Usage

### Output Structure (Kinodynamic Only)
```python
nmpc_joints_acc = LegsAttr(
    FL=np.array([α1, α2, α3]),  # 3 joint accelerations in rad/s²
    FR=np.array([α1, α2, α3]),
    RL=np.array([α1, α2, α3]),
    RR=np.array([α1, α2, α3]),
)
```

### Where it's Used

#### 5.1 Feedforward Acceleration (PRIMARY USE - KINODYNAMIC ONLY)

**File:** `swing_trajectory_controller.py`

**Lines:** 101

**Purpose:** Provide acceleration feedforward to reduce tracking error

```python
def compute_momentum_rate_nondimensional(..., nmpc_joints_acc, ...):
    acceleration = nmpc_joints_acc  # Store for later use
```

Used in feedforward term: `τ = M·a_desired + ...` where `a_desired = nmpc_joints_acc`

**Why it helps:**
- Anticipates required accelerations
- Reduces lag between commanded and actual motion
- Improves tracking performance in feedback linearization

---

## Part 6: best_sample_freq Usage

### Output Structure
```python
best_sample_freq = float  # Optimized step frequency in Hz (e.g., 2.5 Hz)
```

### Where it's Used

#### 6.1 Gait Timing Optimization (PRIMARY USE)

**File:** `wb_interface.py`

**Lines:** 352

**Purpose:** Update step frequency if gait optimization was triggered

```python
if optimize_swing == 1:
    self.pgg.step_freq = np.array([best_sample_freq])[0]  # Update frequency
    self.frg.stance_time = (1 / self.pgg.step_freq) * self.pgg.duty_factor
    swing_period = (1 - self.pgg.duty_factor) * (1 / self.pgg.step_freq)
    self.stc.regenerate_swing_trajectory_generator(
        step_height=self.step_height, 
        swing_period=swing_period
    )
```

**Data Flow:**
```
best_sample_freq from MPC (e.g., 2.51 Hz)
    ↓
Update PeriodicGaitGenerator.step_freq
    ↓
Recompute stance_time = (1/freq) × duty_factor
    ↓
Recompute swing_period = (1 - duty_factor) / freq
    ↓
Regenerate swing trajectories with new timing
    ↓
New gait pattern applied to all legs
```

**Example:**
```
Current frequency: 2.0 Hz → step_period = 0.5 s
Best frequency:   2.5 Hz → step_period = 0.4 s
Duty factor:      0.7

New timing:
- Stance time: (0.4 s) × 0.7 = 0.28 s (was 0.35 s)
- Swing period: (0.4 s) × 0.3 = 0.12 s (was 0.15 s)
```

#### 6.2 Initialization/Fallback (SUPPORTING USE)

**File:** `quadruped_pympc_wrapper.py`

**Lines:** 45

**Purpose:** Initialize or fallback to nominal frequency

```python
self.best_sample_freq = self.wb_interface.pgg.step_freq  # Initial value
```

Used when MPC doesn't optimize frequency (default behavior).

---

## Part 7: nmpc_predicted_state Usage

### Output Structure
```python
nmpc_predicted_state = np.ndarray  # Shape depends on model
# Centroidal: (12,) or (12, horizon) - CoM position/velocity/orientation
# Kinodynamic: (24,) or (24, horizon) - Full state trajectory
```

### Where it's Used

#### 7.1 Commented-Out Position Prediction

**File:** `wb_interface.py`

**Lines:** 421

**Purpose:** Could predict CoM position for IK computation (currently unused)

```python
qpos_predicted = copy.deepcopy(qpos)
# TODO use predicted rotation too
# qpos_predicted[0:3] = nmpc_predicted_state[0:3]  # Not currently used
```

**Why it's commented out:**
- MPC state and robot state have different reference frames
- Direct substitution may cause numerical issues
- Requires careful coordinate frame conversion

---

#### 7.2 Logging/Visualization (SUPPORTING USE)

**Files:**
- `quadruped_pympc_wrapper.py` (line 44, 142, 258)
- Stored for tracking/debugging

**Purpose:** Record MPC's predicted trajectory for analysis

```python
self.nmpc_predicted_state = np.zeros(12)  # Initialize
# Updated every control loop
self.nmpc_predicted_state = self.srbd_controller_interface.compute_control(...)
```

---

## Part 8: Summary Table of All Usages

| Output | Data Type | Primary Use | File | Lines | Update Rate |
|--------|-----------|----------|------|-------|------------|
| **nmpc_GRFs** | LegsAttr(3×4) | Stance torque via J^T@F | wb_interface.py | 366-369 | 100 Hz |
| | | Observation | wrapper.py | 39, 183 | 100 Hz |
| **nmpc_footholds** | LegsAttr(3×4) | Swing trajectory endpoint | wb_interface.py | 389 | 100 Hz |
| | | Stance position hold | wb_interface.py | 400 | 100 Hz |
| | | Early stance detector ref | wb_interface.py | 359 | 100 Hz |
| | | Observation | wrapper.py | 40, 137 | 100 Hz |
| **nmpc_joints_pos** | LegsAttr(3×4) | Joint position reference (kinodynamic) | wb_interface.py | 439 | 100 Hz |
| | | Error computation | stc.py | 95 | 100 Hz |
| **nmpc_joints_vel** | LegsAttr(3×4) | Joint velocity reference (kinodynamic) | wb_interface.py | 440* | 100 Hz |
| | | Velocity error | stc.py | 98 | 100 Hz |
| **nmpc_joints_acc** | LegsAttr(3×4) | Acceleration feedforward | stc.py | 101 | 100 Hz |
| **best_sample_freq** | float | Update gait timing (if optimize_swing==1) | wb_interface.py | 352 | Variable |
| **nmpc_predicted_state** | np.ndarray | CoM prediction (commented) | wb_interface.py | 421 | 100 Hz |
| | | Logging/visualization | wrapper.py | 142, 258 | 100 Hz |

*Bug: Line 440 incorrectly assigns to des_joints_pos instead of des_joints_vel

---

## Part 9: Data Flow Diagram - From SRBD to Actuators

```
┌────────────────────────────────────────────────┐
│ SRBD Controller Interface                      │
│ compute_control() at 100 Hz                    │
└────────┬──────────────────────────────────────┘
         │
    ┌────┴────┬────────────┬───────────┬──────────┐
    │         │            │           │          │
    ▼         ▼            ▼           ▼          ▼
 GRFs   Footholds    Joints*      Frequency   Pred.State
    │         │            │           │          │
    │         │            │           │          └─→ Logging
    │         │            │           │
    │         │            │        optimize_swing==1?
    │         │            │           │         (optional)
    │         │            │           ▼
    │         │            │      Update PGG freq
    │         │            │           │
    │         │            │           ▼
    │         │            │      Regenerate swing period
    │         │            │           │
    ▼         ▼            ▼           │
 WBInterface.compute_stance_and_swing_torque()
    │         │            │           │
    │         ▼            ▼           │
    │    Swing Controller  │           │
    │    (PD tracking)     │           │
    │         │            │           │
    │    touch_down = footholds[leg]   │
    │         │            │           │
    │         ▼            │           │
    │    Swing torques τ_swing          │
    │         │            │           │
    ▼         │            │           │
 Inverse      │            ▼           │
 Dynamics     │        (If kinodynamic)
 τ_stance=    │        des_pos = joints_pos
 -J^T@GRF     │        des_vel = joints_vel
    │         │        acc_ff = joints_acc
    │         │            │
    │         ▼            ▼
    │    IK or Direct Reference
    │         │            │
    │         └────┬────────┘
    │              │
    │              ▼
    │         PD Control
    │         τ = Kp(des-actual) + Kd(ddes-dactual)
    │              │
    └──────────┬───┘
               │
               ▼
         Joint Torques τ (12D)
               │
               ▼
         Simulator/Robot Actuators
```

---

## Part 10: Critical Flow for Uneven Terrain Locomotion

**Current State:**
1. MPC outputs GRFs and footholds
2. These directly control stance and swing
3. No terrain adaptation in outputs
4. Early stance detector provides only reactive collision recovery

**Gap for Uneven Terrain:**
The outputs are terrain-blind. While outputs exist:
- `nmpc_GRFs` - doesn't adapt to slope
- `nmpc_footholds` - don't account for terrain roughness
- `best_sample_freq` - doesn't adapt to stability margin
- `nmpc_predicted_state` - could be used for collision prediction (currently not)

**Improvement Opportunities:**
1. Modify MPC to output terrain-adaptive GRFs (use terrain estimates in cost)
2. Add terrain-aware foothold computation before SRBD interface
3. Use `nmpc_predicted_state` for lookahead collision detection
4. Adapt `best_sample_freq` based on terrain roughness

---

## Summary

The SRBD interface outputs flow directly into the control pipeline:

1. **Immediately Used (100 Hz):**
   - `nmpc_GRFs` → Stance torques (most critical)
   - `nmpc_footholds` → Swing endpoints and stance positions

2. **Conditionally Used:**
   - `nmpc_joints_pos/vel/acc` → Only for kinodynamic models
   - `best_sample_freq` → Only if gait optimization enabled

3. **Currently Unused:**
   - `nmpc_predicted_state` → Logged but not fed back for control

4. **Observation/Analysis:**
   - All outputs available in environment observations for learning

This is where terrain-adaptive improvements must be integrated to enable robust uneven terrain locomotion.
