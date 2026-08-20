# Step Height Adaptation: Numerical Example

## Configuration Parameters Used

From `config.py`:
- **Base step height** (`step_height`): 0.2 × hip_height = **0.104 m** (assuming hip_height = 0.52 m for GO2)
- **Pitch sensitivity** (`swing_pitch_sensitivity`): **0.5**
- **Velocity sensitivity** (`swing_velocity_sensitivity`): **0.6**
- **Velocity max** (`swing_velocity_max`): **1.0 m/s**
- **Step height min** (`swing_height_min`): 0.1 × hip_height = **0.052 m**
- **Step height max** (`swing_height_max`): 0.6 × hip_height = **0.312 m**
- **Obstacle boost increment** (`obstacle_retry_boost_increment`): 0.1 × hip_height = **0.052 m**
- **Terrain pitch**: **0.0 rad** (flat ground)
- **Slowdown ramp rate** (`obstacle_slowdown_ramp_rate`): **0.3 m/s²**

---

## Formulas Used

### 1. Pitch/Velocity Adaptive Step Height

```
pitch_factor = 1.0 + swing_pitch_sensitivity × sin(terrain_pitch)
velocity_normalized = min(velocity / velocity_max, 1.0)
velocity_factor = 1.0 - swing_velocity_sensitivity × velocity_normalized
h_adaptive = h_base × pitch_factor × velocity_factor
h_final = clamp(h_adaptive, h_min, h_max)
```

### 2. Step Height with Obstacle Boost

```
obstacle_boost = min(attempt_count × boost_increment, max_boost)
h_with_boost = h_adaptive + obstacle_boost
h_final = clamp(h_with_boost, h_min, h_max)
```

### 3. Velocity Recovery (Obstacle Slowdown)

```
commanded_velocity_target = 0.0 m/s (or 0.1 m/s based on config)
current_velocity += ramp_rate × dt
Recovery time = (pre_collision_velocity - slowdown_velocity) / ramp_rate
```

---

## Scenario: Robot Accelerates 0→1 m/s, Hits Stair, Recovers 0→1 m/s

### Phase 1: Acceleration to Stair Impact (0.0 → 1.0 m/s)

| Velocity (m/s) | vel_norm | vel_factor | h_base × pitch_factor | h_adaptive (m) | h_final (m) | Notes |
|---|---|---|---|---|---|---|
| **0.0** | 0.000 | 1.000 | 0.104 | **0.104** | **0.104** | Starting at rest |
| **0.1** | 0.100 | 0.940 | 0.104 | **0.098** | **0.098** | Low speed: minimal reduction |
| **0.2** | 0.200 | 0.880 | 0.104 | **0.092** | **0.092** | Gradual reduction |
| **0.3** | 0.300 | 0.820 | 0.104 | **0.085** | **0.085** | Speed increasing |
| **0.4** | 0.400 | 0.760 | 0.104 | **0.079** | **0.079** | Moderate reduction |
| **0.5** | 0.500 | 0.700 | 0.104 | **0.073** | **0.073** | Mid-range |
| **0.6** | 0.600 | 0.640 | 0.104 | **0.067** | **0.067** | Higher speed |
| **0.7** | 0.700 | 0.580 | 0.104 | **0.060** | **0.060** | Approach max |
| **0.8** | 0.800 | 0.520 | 0.104 | **0.054** | **0.054** | Near limit |
| **0.9** | 0.900 | 0.460 | 0.104 | **0.048** | **0.048** | Just above min |
| **1.0** | 1.000 | 0.400 | 0.104 | **0.042** | **0.052** | **Max speed reduction, clamped to h_min** |

**Key Observations:**
- Step height **decreases by 60%** (0.104 → 0.052 m) as velocity increases to 1.0 m/s
- At 1.0 m/s, step height hits the minimum clamp (0.052 m)
- This low step height at high speed makes the robot vulnerable to stairs

---

### Phase 2: Obstacle Impact & Slowdown Triggered (Collision at 1.0 m/s)

**At collision:**
- Robot moving at **1.0 m/s**
- Step height: **0.052 m** (too low for stair)
- EarlyStanceDetector signals collision
- `obstacle_slowdown_active = True`
- `obstacle_slowdown_prev_speed = 1.0 m/s`
- `obstacle_slowdown_current_speed = 0.0 m/s`

**Velocity Recovery Phase** (obstacle slowdown ramp):

| Time (s) | Elapsed | current_velocity (m/s) | velocity_norm | vel_factor | h_adaptive (m) | h_final (m) | Recovery % |
|---|---|---|---|---|---|---|---|
| **0.00** | 0.000 | 0.000 | 0.000 | 1.000 | 0.104 | **0.104** | 0% |
| **0.10** | 0.100 | 0.030 | 0.030 | 0.982 | 0.102 | **0.102** | 3% |
| **0.20** | 0.200 | 0.060 | 0.060 | 0.964 | 0.100 | **0.100** | 6% |
| **0.30** | 0.300 | 0.090 | 0.090 | 0.946 | 0.098 | **0.098** | 9% |
| **0.40** | 0.400 | 0.120 | 0.120 | 0.928 | 0.096 | **0.096** | 12% |
| **0.50** | 0.500 | 0.150 | 0.150 | 0.910 | 0.095 | **0.095** | 15% |
| **1.00** | 1.000 | 0.300 | 0.300 | 0.820 | 0.085 | **0.085** | 30% |
| **1.50** | 1.500 | 0.450 | 0.450 | 0.730 | 0.076 | **0.076** | 45% |
| **2.00** | 2.000 | 0.600 | 0.600 | 0.640 | 0.067 | **0.067** | 60% |
| **2.50** | 2.500 | 0.750 | 0.750 | 0.550 | 0.057 | **0.057** | 75% |
| **3.00** | 3.000 | 0.900 | 0.900 | 0.460 | 0.048 | **0.048** | 90% |
| **3.33** | 3.333 | 1.000 | 1.000 | 0.400 | 0.042 | **0.052** | 100% (recovered) |

**Calculation Note:** Recovery time = (1.0 m/s - 0.0 m/s) / 0.3 m/s² = **3.33 seconds**

---

### Phase 3: Obstacle Retry Boost Effect (During Recovery)

If obstacle retry is enabled and the robot hits the stair again while recovering:

| Attempt # | Boost (m) | h_adaptive (m) | h_with_boost (m) | h_final (m) | Effect |
|---|---|---|---|---|---|
| **0** (no collision) | 0.000 | 0.076 | 0.076 | 0.076 | Normal step height |
| **1st retry** | 0.052 | 0.076 | 0.128 | **0.128** | +41% increase |
| **2nd retry** | 0.104 | 0.076 | 0.180 | **0.180** | +136% increase |
| **3rd retry** | 0.156 | 0.076 | 0.232 | **0.232** | +206% increase |
| **Capped (max boost)** | 0.156 | 0.076 | 0.232 | **0.232** | Max boost reached |

**Key Observation:**
- If robot hits the stair during recovery at 1.5 m/s with step height 0.076 m
- Boost increases from 0 → 0.052 → 0.104 m per retry attempt
- After 3 attempts, step height reaches **0.232 m**, which may clear the stair
- Boost caps at `obstacle_retry_max_boost = 0.156 m`

---

## Summary Table: Full Scenario Flow

| Phase | Velocity (m/s) | Step Height (m) | Mechanism | Duration |
|---|---|---|---|---|
| **Acceleration** | 0.0 → 1.0 | 0.104 → 0.052 | Velocity-adaptive reduction | ~3-5 sec (gait dependent) |
| **Impact** | 1.0 | 0.052 | Collision detected | Instantaneous |
| **Slowdown** | 1.0 → 0.0 | 0.052 | Commanded speed drops | ~0.05 sec |
| **Recovery** | 0.0 → 1.0 | 0.052 → 0.104 | Ramp up at 0.3 m/s² | **3.33 sec** |
| **With Retry Boost** | 1.5 (during recovery) | 0.076 → 0.232 | +52 mm per retry attempt | Per attempt |

---

## Why This Solves High-Speed Stair Problem

### Before (No Slowdown):
1. Robot accelerates to 1.0 m/s → step height drops to 0.052 m
2. Hits stair → **collision at low step height, no time to react**
3. Robot fails/falls

### After (With Slowdown):
1. Robot accelerates to 1.0 m/s → step height drops to 0.052 m
2. Hits stair → collision detected
3. **Speed immediately drops to 0.0 m/s** → step height returns to **0.104 m**
4. With higher step height, robot can clear the stair obstacle
5. If it hits again during recovery, retry boost adds **+52 mm per attempt**
6. Speed gradually recovers over 3.33 seconds

### Alternative Scenario with Faster Ramp:
If `obstacle_slowdown_ramp_rate = 0.5 m/s²` instead of 0.3 m/s²:
- Recovery time: (1.0 - 0.0) / 0.5 = **2.0 seconds** (1.3 sec faster)
- Step height trajectory follows same curve, just compressed in time

---

## Parameter Sensitivity Analysis

### Effect of Velocity Sensitivity

| swing_velocity_sensitivity | h at 0.5 m/s | h at 1.0 m/s | Range | Aggressiveness |
|---|---|---|---|---|
| **0.3** (low) | 0.078 m | 0.072 m | 0.006 m | Gentle reduction |
| **0.6** (current) | 0.062 m | 0.052 m | 0.012 m | Moderate |
| **0.8** (high) | 0.048 m | 0.042 m | 0.016 m | Aggressive reduction |
| **1.0** (max) | 0.052 m | 0.042 m | 0.010 m | Max reduction |

**Current setting (0.6)** provides good balance between speed stability and obstacle clearance.

---

## Configuration Tuning Recommendations

### To Reduce Recovery Time:
```python
'obstacle_slowdown_ramp_rate': 0.5  # 0.3 → 0.5 m/s² for faster recovery
# Recovery time: 3.33 s → 2.0 s
```

### To Increase Step Height at Speed:
```python
'swing_velocity_sensitivity': 0.4  # 0.6 → 0.4 to reduce velocity penalty
# h at 1.0 m/s: 0.052 m → 0.062 m (+19%)
```

### To Be More Aggressive on Obstacles:
```python
'obstacle_retry_boost_increment': 0.08  # 0.052 m → 0.08 m per attempt
# 3rd attempt: 0.232 m → 0.292 m (+25%)
```

### To Handle Steeper Stairs:
```python
'swing_height_max': 0.4 * hip_height  # 0.312 m → 0.208 m max
# Allows boost to reach higher effective heights
```

---

## Key Takeaways

1. **Velocity Reduction is Aggressive**: Step height drops to 40% at 1.0 m/s (velocity_sensitivity=0.6)
2. **Slowdown Triggers Recovery**: Collision immediately returns step height to 100%, giving robot time to adjust
3. **Recovery is Gradual**: 3.33-second ramp at 0.3 m/s² allows step height to increase as velocity increases
4. **Retry Boost Stacks**: Each collision attempt adds +52 mm, stacking up to +156 mm max boost
5. **The Sweet Spot**: Combination of slowdown + retry boost handles stairs that would fail at constant high speed
