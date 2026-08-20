# Swing Generators - Quick Reference Guide

## Comparison Summary

### Scipy Implementation (Recommended)
```
Input:
  ├─ lift_off: [0.1, 0.15, 0.0]
  ├─ touch_down: [0.25, 0.15, 0.0]
  ├─ swing_time: 0.125s (at apex)
  └─ step_height: 0.06m

Processing:
  ├─ Create 5 control points (lift, 2 mid, 2 near touchdown)
  ├─ CubicSpline interpolation (C² continuous)
  ├─ Evaluate splines: position, velocity, acceleration
  └─ Derivatives computed exactly via scipy

Output:
  ├─ position: [0.175, 0.15, 0.045]
  ├─ velocity: [0.70, 0.00, 0.60]
  └─ acceleration: [0.40, 0.00, 0.20]

PD Control:
  ├─ error_pos = desired - actual
  ├─ error_vel = desired - actual
  ├─ acceleration = des_acc + Kp*e_pos + Kd*e_vel
  └─ τ = J.T @ F_pd + M @ J^{-T} @ (a - J_dot*q_dot) + h

Result:
  └─ τ_swing: [1.2, -0.8, 0.5] Nm (joint torques for leg)
```

### Explicit Bézier Implementation (Faster)
```
Input:
  ├─ lift_off: [0.1, 0.15, 0.0]
  ├─ touch_down: [0.25, 0.15, 0.0]
  ├─ swing_time: 0.125s
  └─ step_height: 0.06m

Processing:
  ├─ Select 4 control points (based on phase)
  ├─ Phase 1 (t ≤ T/2): Low → High to apex
  ├─ Phase 2 (t > T/2): Apex → Low to touchdown
  ├─ Cubic Bézier basis functions
  └─ Analytical derivatives (closed-form)

Bézier Formula:
  B(t) = (1-t)³P₀ + 3t(1-t)²P₁ + 3t²(1-t)P₂ + t³P₃

Output:
  ├─ position: via B(t) evaluation
  ├─ velocity: via B'(t) = 3(1-t)²(P₁-P₀) + ...
  └─ acceleration: via B''(t) = 6(1-t)(P₂-2P₁+P₀) + ...

Same PD control & feedback linearization
```

---

## Architecture Map

```
quadruped_pympc/helpers/swing_generators/
│
├─ scipy_swing_trajectory_generator.py
│  └─ SwingTrajectoryGenerator
│     ├─ __init__()
│     ├─ createCurve()           [Generate 5-point spline]
│     ├─ compute_trajectory_references()  [Evaluate @ swing_time]
│     └─ plot_trajectory_3d()    [Visualization]
│
└─ explicit_swing_trajectory_generator.py
   └─ SwingTrajectoryGenerator
      ├─ __init__()
      ├─ compute_control_points() [Select 4-point Bézier]
      ├─ compute_trajectory_references() [Evaluate Bézier]
      └─ plot_trajectory_3d()    [Visualization]

Called by:
  └─ swing_trajectory_controller.py
     └─ SwingTrajectoryController
        ├─ __init__() [Select generator from config]
        ├─ regenerate_swing_trajectory_generator() [Update timing]
        ├─ compute_swing_control_cartesian_space() [Compute τ]
        ├─ update_swing_time() [Increment timer]
        └─ check_*_condition() [Gait phase detection]

Integrated by:
  └─ interfaces/wb_interface.py
     └─ WBInterface.compute_stance_and_swing_torque()
        └─ For each leg in swing:
           ├─ Update swing_time
           ├─ Get trajectory reference
           ├─ Compute PD errors
           ├─ Apply feedback linearization
           └─ Return joint torques
```

---

## Control Flow: One Control Cycle

```
[t = 0.002s cycle]

1. GET CURRENT STATE
   ├─ feet_pos: [FL_pos, FR_pos, RL_pos, RR_pos] (world frame)
   ├─ feet_vel: [FL_vel, FR_vel, RL_vel, RR_vel]
   ├─ feet_jac: [FL_jac, FR_jac, RL_jac, RR_jac]
   ├─ contact_state: [1, 0, 0, 1]  (FL/RR in stance, FR/RL in swing)
   └─ swing_time: [0.0, 0.100, 0.0, 0.102]  (s)

2. UPDATE SWING TIMER
   For each leg:
     if contact[leg] == 0:
       swing_time[leg] += 0.002

3. FOR EACH LEG IN SWING (FR at leg_id=1, RL at leg_id=2)
   ┌─ LEG FR (leg_id=1)
   │  ├─ swing_time[1] = 0.102s
   │  ├─ lift_off = [0.1, 0.15, 0.0]
   │  ├─ touchdown = [0.25, 0.15, 0.0]
   │  │
   │  ├─ Call: swing_generator.compute_trajectory_references()
   │  │  ├─ Create spline with control points
   │  │  ├─ Evaluate @ t=0.102s
   │  │  └─ Return: des_pos, des_vel, des_acc
   │  │
   │  ├─ Compute errors
   │  │  ├─ err_pos = des_pos - feet_pos[FR]
   │  │  └─ err_vel = des_vel - feet_vel[FR]
   │  │
   │  ├─ Compute swing control
   │  │  └─ τ[FR] = compute_swing_control_cartesian_space()
   │  │
   │  └─ Store: des_foot_pos[FR] = des_pos
   │
   └─ LEG RL (leg_id=2)
      └─ [Same process as FR]

4. STANCE LEGS (FL, RR) in contact
   └─ τ[leg] = -J.T @ nmpc_GRFs[leg]

5. FRICTION COMPENSATION
   For each leg:
     τ[leg] -= passive_forces[leg]

6. OUTPUT
   ├─ τ_FL: [-0.2, 0.1, 0.3] Nm
   ├─ τ_FR: [1.2, -0.8, 0.5] Nm  (swing)
   ├─ τ_RL: [0.8, -0.9, 0.4] Nm  (swing)
   └─ τ_RR: [-0.1, 0.2, 0.2] Nm

7. SEND TO ACTUATORS
   └─ env.step(action=τ)
```

---

## Trajectory Shape During Swing

### Scipy (5-point CubicSpline)
```
Height (Z)
  │     ╱╲
  │    ╱  ╲      Control points: •
  │   •    •     Trajectory: ——
  │  ╱      ╲
  │ •        •
  │╱          ╲
  └─────────────────► Time
  0    T/4   T/2  3T/4  T

Position Smoothness:
  • Position:     C⁰ (continuous)
  • Velocity:     C¹ (smooth)
  • Acceleration: C² (very smooth)
```

### Explicit (2-phase Bézier)
```
Height (Z)
  │      ╱╲
  │     ╱  ╲      Phase 1:  Accelerate to apex
  │    ╱    ╲     Phase 2:  Decelerate to landing
  │   ╱      ╲
  │  ╱        ╲
  │╱          ╲
  └─────────────────► Time
  0    T/2         T

Position Smoothness:
  • Position:     C⁰ (continuous)
  • Velocity:     C¹ (continuous)
  • Acceleration: ⚠ C⁰ (discontinuous at T/2)
```

---

## Key Variables Reference

### SwingTrajectoryGenerator
```python
# Scipy
self._curve_x          # CubicSpline object for x-position
self._curve_x_vel      # 1st derivative (velocity)
self._curve_x_acc      # 2nd derivative (acceleration)
self.step_height       # Max swing height (m)
self.swing_period      # Total swing duration (s)

# Explicit
# [No spline storage - computed on-the-fly]
```

### SwingTrajectoryController
```python
self.swing_generator        # Selected generator ('scipy' or 'explicit')
self.swing_time             # [FL, FR, RL, RR] time in swing (s)
self.swing_period           # Total duration
self.position_gain_fb       # Kp for PD control (typ. 500)
self.velocity_gain_fb       # Kd for PD control (typ. 10)
self.use_feedback_linearization  # Enable acceleration tracking
```

### WBInterface
```python
self.stc                # SwingTrajectoryController instance
self.stc.swing_time[leg_id]  # Current swing time for leg
self.frg.lift_off_positions[leg_name]   # Where leg started swing
self.esd.hitpoints[leg_name]            # Collision point (reflex)
self.esd.hitmoments[leg_name]           # Collision time (reflex)
```

---

## Configuration Parameters

```python
# In config.py simulation_params:

'swing_generator': 'scipy'              # or 'explicit'
'swing_position_gain_fb': 500           # PD Kp
'swing_velocity_gain_fb': 10            # PD Kd
'step_height': 0.2 * hip_height         # Normal swing height (m)
'reflex_max_step_height': 0.8 * hip_height  # Reflex swing height (m)
'reflex_trigger_mode': 'tracking'       # or 'geom_contact' or False
'reflex_next_steps_height_enhancement': True   # Activate after reflex

# In config.py mpc_params:
'optimize_step_freq': True              # Enable gait optimization
```

---

## Common Call Stacks

### Normal Swing Control (Every 2ms)
```
simulation.py: env.step()
  └─ wb_interface.compute_stance_and_swing_torque()
     └─ For each leg in swing:
        └─ stc.compute_swing_control_cartesian_space()
           ├─ stc.swing_generator.compute_trajectory_references()
           │  └─ scipy: evaluate _curve_x(t), _curve_x_vel(t), etc.
           │  └─ explicit: compute Bézier basis & derivatives
           ├─ Compute PD errors
           └─ Apply feedback linearization
```

### Gait Optimization (Every swing cycle, ~1 second)
```
wb_interface.check_touch_down_condition() returns 1
  └─ compute_stance_and_swing_torque(): optimize_swing == 1
     └─ stc.regenerate_swing_trajectory_generator(new_step_height, new_swing_period)
        └─ Create new generator with updated parameters
```

### Early Stance Reflex (Triggered on collision)
```
early_stance_detector.update_detection() detects obstacle
  └─ Sets: esd.hitpoints[leg], esd.hitmoments[leg]
     └─ Sets: stc.swing_generator.reflex_next_steps_height_enhancement = True
        └─ Next swing cycle:
           └─ compute_trajectory_references(early_stance_hitmoment, early_stance_hitpoint)
              └─ createCurve() regenerates trajectory from collision point
```

---

## Debugging Checklist

- [ ] Check config.py has correct swing_generator selected
- [ ] Verify swing_position_gain_fb and swing_velocity_gain_fb values
- [ ] Confirm step_height is reasonable (0.04-0.1m for Go2)
- [ ] Check if early_stance_detector is enabled
- [ ] Verify compute_trajectory_references() returns expected values
- [ ] Check Jacobian accuracy (feeds into τ computation)
- [ ] Confirm mass_matrix is properly computed
- [ ] Verify feedback_linearization is enabled
- [ ] Check feet_jac_dot is accurate (finite difference may have noise)

---

## Performance Tips

1. **If swing control is slow:**
   - Switch to 'explicit' generator (faster computation)
   - Reduce control frequency if possible
   - Pre-allocate arrays in loops

2. **If foot tracking is poor:**
   - Increase Kp (position_gain_fb)
   - Verify Jacobian accuracy
   - Check for numerical issues (small J determinant)

3. **If foot overshoots target:**
   - Increase Kd (velocity_gain_fb)
   - Reduce Kp slightly
   - Verify step_height is appropriate

4. **If reflex not triggering:**
   - Check reflex_trigger_mode is not False
   - Lower relative_tracking_error_threshold
   - Verify contact sensors are working

---

**This quick reference complements the detailed SWING_GENERATORS_DOCUMENTATION.md**
