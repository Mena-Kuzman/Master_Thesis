# Swing Generator Input Sources: lift_off and touch_down

## Overview

The explicit swing generator receives `lift_off` and `touch_down` points as inputs to its `compute_trajectory_references()` method. These points define the starting and ending positions of a foot swing phase.

```
lift_off  ────────────► Explicit Swing Generator ──────────► Trajectory (pos, vel, acc)
                              │
touch_down ───────────────────┘
```

---

## Data Flow: Where Points Come From

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CONTROL HIERARCHY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  MPC CONTROLLER (100 Hz)                                                     │
│  ├─ Optimizes footholds & GRF over horizon                                  │
│  └─ Output: nmpc_footholds[leg] = desired touchdown position                │
│                                                                               │
│                ▼                                                              │
│                                                                               │
│  FOOTHOLD REFERENCE GENERATOR                                               │
│  ├─ Tracks current feet positions (lift_off_positions)                      │
│  ├─ Updated continuously during stance                                      │
│  └─ Output: self.frg.lift_off_positions[leg] = current foot position        │
│                                                                               │
│                ▼                                                              │
│                                                                               │
│  WB INTERFACE - compute_swing_control_cartesian_space()                      │
│  ├─ Assembles inputs for swing control                                       │
│  ├─ Parameters:                                                              │
│  │  ├─ lift_off = self.frg.lift_off_positions[leg_name]                    │
│  │  ├─ touch_down = nmpc_footholds[leg_name]                               │
│  │  └─ [other: foot_pos, foot_vel, jacobian, etc.]                         │
│  │                                                                           │
│  │                ▼                                                          │
│  │                                                                           │
│  │  SWING TRAJECTORY CONTROLLER                                             │
│  │  ├─ Calls: swing_generator.compute_trajectory_references(                │
│  │  │           swing_time, lift_off, touch_down, ...)                     │
│  │  │                                                                       │
│  │  │                ▼                                                      │
│  │  │                                                                       │
│  │  │  EXPLICIT SWING GENERATOR                                            │
│  │  │  ├─ Receives: lift_off, touch_down                                   │
│  │  │  ├─ Creates: 4-point Bézier curve                                    │
│  │  │  └─ Returns: [desired_pos, desired_vel, desired_acc]                 │
│  │  │                                                                       │
│  │  └─ PD Control + Feedback Linearization                                 │
│  │     └─ Output: Joint torques τ                                          │
│  │                                                                          │
│  └─ Output: τ[leg] ──► Actuators                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Input #1: `lift_off` - Current Foot Position

### Source
```python
# In wb_interface.py, line 389:
lift_off = self.frg.lift_off_positions[leg_name]
```

### What It Is
The **current position of the foot at the start of swing phase** (when it lifts off from the ground).

### How It's Tracked

```python
# In FootholdReferenceGenerator.__init__() (foothold_reference_generator.py):
self.lift_off_positions = copy.deepcopy(lift_off_positions)  # Initial feet positions
```

```python
# Updated during stance phase when foot is in contact
# (Tracked implicitly as foot position state)
```

### Example Values
```python
# For a quadruped robot (e.g., Unitree Go2):
lift_off['FL'] = np.array([0.18, 0.12, -0.18])    # Front-left foot position
lift_off['FR'] = np.array([0.18, -0.12, -0.18])   # Front-right foot position  
lift_off['RL'] = np.array([-0.18, 0.12, -0.18])   # Rear-left foot position
lift_off['RR'] = np.array([-0.18, -0.12, -0.18])  # Rear-right foot position

# All in world frame [x, y, z]
# z is typically negative (feet below body)
```

### Key Insight
`lift_off` is **essentially the current foot position** at the moment swing phase begins. It represents where the foot is when it loses contact with the ground.

---

## Input #2: `touch_down` - Desired Foot Position

### Source
```python
# In wb_interface.py, line 389:
touch_down = nmpc_footholds[leg_name]
```

### What It Is
The **desired foot position at the end of swing phase** (touchdown target), computed by the **MPC optimizer**.

### How It's Generated

```python
┌──────────────────────────────────────────────────────┐
│ NONLINEAR MPC CONTROLLER                             │
│ ├─ Optimizes motion over prediction horizon         │
│ ├─ Variables: base state, GRF, footholds            │
│ └─ Returns:                                          │
│    ├─ nmpc_GRFs[leg] (ground reaction forces)       │
│    ├─ nmpc_footholds[leg] ◄─── Used as touch_down  │
│    └─ nmpc_predicted_state (trajectory)             │
└──────────────────────────────────────────────────────┘
```

From `simulation/simulation.py`:
```python
# Line 120: MPC outputs include nmpc_footholds
"nmpc_footholds",

# Line 276: Passed to wb_interface
nmpc_footholds=ctrl_state["nmpc_footholds"]
```

### Example Values
```python
# MPC-optimized footholds:
touch_down['FL'] = np.array([0.25, 0.14, -0.18])    # Next FL foot landing
touch_down['FR'] = np.array([0.25, -0.14, -0.18])   # Next FR foot landing
touch_down['RL'] = np.array([-0.10, 0.12, -0.18])   # Next RL foot landing
touch_down['RR'] = np.array([-0.10, -0.12, -0.18])  # Next RR foot landing

# All in world frame [x, y, z]
# Typically ~0.05-0.1m forward of lift_off in x-direction
```

### Key Insight
`touch_down` is the **MPC-optimized target** for where the foot should land. It encodes the planned locomotion trajectory and responds to:
- Desired base velocity
- Terrain geometry (if VFA is enabled)
- Gait pattern (duty factor, step frequency)
- Balance constraints

---

## Swing Phase Trajectory Construction

Once the explicit swing generator receives these two points:

```python
# In explicit_swing_trajectory_generator.py:

def compute_trajectory_references(
    self, swing_time, lift_off, touch_down, ...
):
    # Step 1: Compute 4 control points
    bezier_swing_time, cp1, cp2, cp3, cp4 = self.compute_control_points(
        swing_time, lift_off, touch_down
    )
    
    # Control points:
    # cp1 = lift_off                           # Start at current foot position
    # cp2 = lift_off (phase 1) or midpoint     # Intermediate point
    # cp3 = midpoint (phase 1) or touch_down   # Intermediate/end
    # cp4 = touch_down                         # End at desired landing

    # Step 2: Evaluate cubic Bézier curve
    desired_foot_position = (
        (1-t)³*cp1 + 3t(1-t)²*cp2 + 3t²(1-t)*cp3 + t³*cp4
    )
    
    # Returns smooth trajectory from lift_off ────► touch_down
    return desired_foot_position, desired_foot_velocity, desired_foot_acceleration
```

### Visualized Swing Trajectory

```
TIME: 0 ─────────────────────── 0.5s (swing_period = 1s)
      │                            │
      │                            │
HEIGHT: APEX                       │
      │  ╱╲                        │
      │ ╱  ╲                       │
      │╱____╲                      │
z     │ │   │                      │
      │ │   │                      │
      └─┴───┴──────────────────────

      lift_off                  touch_down
      (Start)                   (End)
      
      ├─── Horizontal distance ─────┤
         x(touch_down) - x(lift_off)
         ≈ 0.05-0.1m forward
```

---

## Lifecycle: How Points Change Over Time

```
SWING CYCLE PROGRESSION:

┌─ Cycle 1 ──────────────────────────────┐
│                                        │
│ STANCE (FL in contact)                 │
│ ├─ lift_off_positions[FL] stays fixed  │
│ ├─ But other legs move...              │
│ └─ When FR starts swing:               │
│    ├─ lift_off[FR] = current pos of FR │
│    └─ touch_down[FR] = MPC optimized   │
│                                        │
│ SWING (FL lifts off → lands)           │
│ ├─ Trajectory: lift_off[FL] ──► touch_down[FL]
│ ├─ At touchdown:                       │
│ │  └─ lift_off_positions[FL] updated to
│ │     touch_down[FL]                   │
│ └─ Now FL in stance, RL starts swing   │
│                                        │
└────────────────────────────────────────┘
```

---

## Special Case: Early Stance Reflex

If obstacle detected during swing:

```python
# In early_stance_detector.py:
if collision_detected:
    hitmoment = current_swing_time
    hitpoint = foot_position_at_collision
    
# Swing generator regenerates trajectory:
# From: lift_off ──┐
#                  ├──► touch_down (original)
#                  ├──► COLLISION POINT
#                  └──► NEW trajectory regenerated
#                       from hitpoint to touch_down
#
# Result: Higher step height to clear obstacle
```

---

## Summary Table

```
┌─────────────────────────────────────────────────────────────┐
│ Parameter      │ Source           │ Updates   │ Typical Value│
├─────────────────────────────────────────────────────────────┤
│ lift_off       │ Current foot pos │ Per swing │ [0.18, 0.12, -0.18]
│                │ (state-based)    │ start     │ m            │
├─────────────────────────────────────────────────────────────┤
│ touch_down     │ MPC optimizer    │ Per MPC   │ [0.25, 0.14, -0.18]
│                │ (horizon-based)  │ update    │ m            │
│                │                  │ (100 Hz)  │              │
├─────────────────────────────────────────────────────────────┤
│ Difference     │ Velocity × time  │ Per cycle │ Δx ≈ 0.05-0.1m
│ (horizontal)   │ × duty factor    │           │ (forward)    │
└─────────────────────────────────────────────────────────────┘
```

---

## Code References

**Explicit Swing Generator Receives Inputs:**
- File: `quadruped_pympc/helpers/swing_generators/explicit_swing_trajectory_generator.py`
- Method: `compute_trajectory_references(swing_time, lift_off, touch_down, ...)`
- Line: ~120

**Inputs Called From:**
- File: `quadruped_pympc/helpers/swing_trajectory_controller.py`
- Method: `compute_swing_control_cartesian_space()`
- Line: ~70

**Inputs Assembled In:**
- File: `quadruped_pympc/interfaces/wb_interface.py`
- Method: `compute_swing_control_cartesian_space()`
- Lines: 383-390

**lift_off_positions Maintained By:**
- File: `quadruped_pympc/helpers/foothold_reference_generator.py`
- Class: `FootholdReferenceGenerator`
- Line: ~23

**touch_down (nmpc_footholds) Generated By:**
- File: `quadruped_pympc/interfaces/srbd_controller_interface.py`
- Method: `compute_control()`
- Line: ~210

---

## Key Insights

1. **lift_off is STATE**: It's the current foot position when swing starts (read from simulator state)

2. **touch_down is PLANNED**: It's the MPC-computed target, based on optimization over the prediction horizon

3. **Separation enables reactive control**: The swing trajectory always starts from current foot position (lift_off) and aims for the MPC-planned target (touch_down), allowing for real-time adjustments

4. **Both updated at different rates**:
   - lift_off: Changes only at swing phase initiation (gait-frequency dependent)
   - touch_down: Updated continuously by MPC at 100 Hz

5. **Reflex integration**: Early stance detector can modify touch_down during swing for obstacle avoidance

