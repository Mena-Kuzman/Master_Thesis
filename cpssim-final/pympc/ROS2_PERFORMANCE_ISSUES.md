# ROS2 Adaptive Gait Performance Issues - Root Cause Analysis

## Critical Issues Found

### 1. **MAJOR: Stale Velocity Data in Mujoco Model**
**Location:** `ros2/run_controller_mujoco.py`, lines 365-377

**Issue:**
In the ROS2 controller, the velocity used by `env.base_lin_vel()` is computed from the **mujoco simulation state**, NOT from the actual robot state received via ROS2 callbacks.

```python
# ROS2 callback updates these:
self.linear_velocity = np.array(msg.linear_velocity)
self.angular_velocity = np.array(msg.angular_velocity)

# Then control loop does:
self.env.mjData.qvel[0:3] = copy.deepcopy(self.linear_velocity)  # Line 371
mujoco.mj_forward(self.env.mjModel, self.env.mjData)  # Line 375

# But then later uses:
base_lin_vel = self.env.base_lin_vel(frame='world')  # Line 385
```

**Why this is a problem:**
- In `simulation.py`, velocity comes directly from the environment: `base_lin_vel = env.base_lin_vel(frame="world")`
- This is the **current, real-time** velocity from the physics simulation step
- In ROS2, the velocity comes from **asynchronous callbacks** that may be stale or out-of-sync with the scheduler
- The velocity is set into mujoco AFTER receiving a message, but the `compute_control_callback` may run multiple times between messages
- This creates **velocity discontinuities** where multiple control loops run with the same old velocity until a new message arrives

### 2. **Velocity History Accumulation Mismatch**
**Location:** `quadruped_pympc/helpers/foothold_reference_generator.py`, lines 17-18, 63-66

**Issue:**
The adaptive gait uses a moving average of the last 20 velocity measurements:

```python
self.adaptive_vel_hist = collections.deque(maxlen=vel_moving_average_length)  # maxlen=20
```

In `simulation.py`, this fills every control loop iteration with fresh velocity data (250 Hz = 0.004s apart).

In ROS2:
- State messages may arrive at **different frequencies** than the 250 Hz control loop
- The velocity history buffer may be **dominated by old measurements** if messages arrive slowly
- Example: If state arrives at 100 Hz but control runs at 250 Hz:
  - Frames 1-2.5 on average use the same 20-measurement velocity history
  - This creates a **persistent velocity lag** of up to 80ms (20 measurements × 0.004s assuming fast updates)

### 3. **Callback Timing and State Synchronization**
**Location:** `ros2/run_controller_mujoco.py`, lines 272-290, 330, 345-355

**Issue:**
The ROS2 node uses asynchronous callbacks with a scheduler:

```python
self.subscription_base_state = self.create_subscription(
    BaseStateMsg, "/dls2/base_state", self.get_base_state_callback, 1  # Queue size = 1
)

if(USE_SCHEDULER):
    self.timer = self.create_timer(1.0/SCHEDULER_FREQ, self.compute_control_callback)
```

Problems:
- Queue size = 1 means **old messages are dropped** if not processed
- If a callback arrives while `compute_control_callback` is running, it overwrites `self.linear_velocity`
- The control loop may see a velocity that was updated **while calculations were in progress**
- **Race condition** between callback writes and control loop reads (not thread-safe without locks)

### 4. **Velocity History Not Cleared Between State Updates**
**Location:** `quadruped_pympc/helpers/foothold_reference_generator.py`, line 17

**Issue:**
The velocity history buffer is never cleared or reset. In simulation, this works because:
- Velocity updates continuously and smoothly
- Old measurements get naturally pushed out as new ones arrive

In ROS2:
- If a state message is late/dropped, the old velocity stays in the buffer
- The buffer contains a **mix of old and new velocities** if update frequency is inconsistent
- This biases `adaptive_speed()` calculation toward past velocities

### 5. **No Velocity Validation Before Using**
**Location:** `quadruped_pympc/helpers/foothold_reference_generator.py`, lines 63-66

**Issue:**
`adaptive_speed()` doesn't check if the velocity measurement is fresh or stale:

```python
def adaptive_speed(self, base_lin_vel_h: np.ndarray, ref_base_lin_vel_h: np.ndarray) -> float:
    self.adaptive_vel_hist.append(base_lin_vel_h)
    velocity_avg = np.mean(list(self.adaptive_vel_hist), axis=0)
    # ... doesn't validate if this is recent data
```

The moving average could contain measurements from up to 80ms ago, causing the adaptive gait to react **too slowly** to velocity changes.

---

## Expected Symptom Confirmation

These issues would cause:
1. **Lower average velocity** - The robot can't reach 2.0 m/s because adaptive stride is computed from stale/averaged velocity
2. **Sluggish acceleration** - Large delays between velocity command and gait adaptation
3. **Oscillations or instability** - Stale velocity spikes cause sudden stride/duty changes
4. **Worse at higher frequencies** - The mismatch between ROS2 message frequency and 250 Hz control loop is more pronounced

---

## Solutions to Try

1. **Use Fresh Velocity Every Loop** (PRIORITY 1)
   - Don't use `self.linear_velocity` from callbacks in the control loop
   - Instead, compute `base_lin_vel = self.env.base_lin_vel(frame='world')` like in `simulation.py`
   - This ensures velocity is always **current with the mujoco state**

2. **Clear Velocity History on Major Gaps** (PRIORITY 2)
   - Track timestamp of last velocity update
   - If gap > 50ms, partially clear the buffer to weight recent data higher

3. **Add Thread-Safe State Handling** (PRIORITY 2)
   - Use locks when accessing `self.linear_velocity` in callbacks vs. control loop
   - Or use a double-buffer pattern

4. **Timestamp-Based Velocity Weighting** (PRIORITY 3)
   - Weight velocity measurements in the moving average by how recent they are
   - Recent measurements count more than old ones

5. **Sync Control Loop to State Messages** (PRIORITY 3)
   - Option: Trigger control loop from `get_base_state_callback` instead of fixed timer
   - Ensures you always have fresh state when computing control
