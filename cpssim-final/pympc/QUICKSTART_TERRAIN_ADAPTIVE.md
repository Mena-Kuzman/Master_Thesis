# Terrain-Adaptive Step Height - Quick Start Guide

## 🚀 TL;DR - Get Started in 30 Seconds

### Enable the Feature
```python
# In your simulation config or at runtime:
from quadruped_pympc import config as cfg
cfg.simulation_params['terrain_adaptive_step_height_enabled'] = True
```

### That's It!
The adaptive step height will now automatically adjust based on:
- **Terrain slope** (upslope = taller steps, downslope = shorter steps)
- **Terrain roughness** (rough = taller steps for obstacle clearance)
- **Locomotion speed** (faster = taller steps for stability margin)

---

## 📊 What It Does

The quadruped's step height changes dynamically:

| Scenario | Step Height | Adaptation |
|----------|-------------|-----------|
| Flat, smooth, slow | 0.056 m | ✓ Baseline |
| Upslope (+15°) | 0.063 m | ⬆️ +13% |
| Downslope (-15°) | 0.049 m | ⬇️ -13% |
| Rough terrain | 0.078 m | ⬆️ +40% |
| High speed (2 m/s) | 0.073 m | ⬆️ +30% |
| Combined all factors | 0.127 m | ⬆️ +126% |

---

## ⚙️ Configuration Parameters

All parameters are in `quadruped_pympc/config.py`:

```python
'terrain_adaptive_step_height_enabled': False  # Toggle feature on/off
'swing_height_min': 0.04                        # Minimum height (m)
'swing_height_max': 0.15                        # Maximum height (m)
'swing_pitch_sensitivity': 0.5                  # Slope sensitivity (0-1)
'swing_roughness_sensitivity': 0.8              # Roughness sensitivity (0-1)
'swing_velocity_sensitivity': 0.3               # Speed sensitivity (0-1)
```

### Tuning Tips
- **Increase `swing_pitch_sensitivity`** if quadruped struggles on slopes
- **Increase `swing_roughness_sensitivity`** if stepping over obstacles fails
- **Increase `swing_velocity_sensitivity`** if gait becomes unstable at high speeds
- **Adjust height limits** if you have a different quadruped model

---

## 🧪 Run the Tests

Verify everything works:
```bash
cd /home/mena/repos/pympc
python test_terrain_adaptive_step_height.py
```

Expected output:
```
============================================================
TEST SUMMARY
============================================================
Test 1 (Roughness Estimator): PASS ✓
Test 2 (Adaptive Step Height): PASS ✓
Test 3 (Configuration): PASS ✓

Overall: ALL TESTS PASS ✓
============================================================
```

---

## 📚 Implementation Details

### Files Added/Modified
1. **NEW:** `quadruped_pympc/helpers/terrain_roughness_estimator.py`
   - Estimates terrain roughness from GRF variance
   
2. **MODIFIED:** `quadruped_pympc/helpers/swing_trajectory_controller.py`
   - Added `compute_terrain_adaptive_step_height()` method
   
3. **MODIFIED:** `quadruped_pympc/config.py`
   - Added 6 new configuration parameters
   
4. **MODIFIED:** `quadruped_pympc/interfaces/wb_interface.py`
   - Integrated adaptive step height into control loop

### Mathematical Model
```
h_adaptive = h_base × pitch_factor × roughness_factor × velocity_factor

pitch_factor = 1.0 + 0.5 × sin(θ_pitch)      [range: 0.5-1.5]
roughness_factor = 1.0 + 0.8 × σ_roughness   [range: 1.0-1.8]
velocity_factor = 1.0 + 0.3 × (v/v_max)      [range: 1.0-1.3]

Final: h_final = clip(h_adaptive, 0.04, 0.15) meters
```

---

## 🔧 How It Works (Simple Version)

1. **Sensor Data:**
   - Terrain pitch/roll from foot positions (already implemented)
   - GRF variance during stance phase (new: roughness estimator)
   - Robot velocity from IMU (already implemented)

2. **Computation:**
   - Compute 3 factors: pitch, roughness, velocity
   - Multiply them together with base step height
   - Clamp result to safe limits [0.04m, 0.15m]

3. **Output:**
   - New step height → Swing trajectory regenerated
   - Next swing phase uses adaptive height
   - Real-time adjustment every control loop

---

## 🎯 Expected Improvements

With this feature enabled, the quadruped should:
- ✅ Better climb slopes without dragging feet
- ✅ Improve obstacle avoidance on rough terrain
- ✅ More stable at higher speeds
- ✅ Adapt to varied terrain without manual tuning
- ✅ Reduced foot collisions with terrain

---

## ⚠️ Safety Notes

- **Feature is disabled by default** for safe testing
- Step height is always clamped to [0.04m, 0.15m]
- Works with existing gait parameters
- Backward compatible (no code changes required to disable)

---

## 🐛 Troubleshooting

**Q: Feature doesn't seem to work?**
A: Check that `terrain_adaptive_step_height_enabled = True` in config

**Q: Steps are too high or too low?**
A: Adjust the sensitivity coefficients (0.5, 0.8, 0.3) in config

**Q: Gait becomes unstable?**
A: Reduce `swing_velocity_sensitivity` or lower `swing_height_max`

**Q: How do I disable it?**
A: Set `terrain_adaptive_step_height_enabled = False` or comment out the integration

---

## 📖 Full Documentation

For detailed mathematical relationships and theory:
- `TERRAIN_ADAPTIVE_STEP_HEIGHT_GUIDE.md` - Comprehensive guide
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `test_terrain_adaptive_step_height.py` - Working examples

---

## 📞 Quick Reference

### Enable
```python
cfg.simulation_params['terrain_adaptive_step_height_enabled'] = True
```

### Check Status
```python
print(cfg.simulation_params.get('terrain_adaptive_step_height_enabled'))
```

### Adjust Sensitivity
```python
cfg.simulation_params['swing_pitch_sensitivity'] = 0.7  # More sensitive to slopes
cfg.simulation_params['swing_roughness_sensitivity'] = 1.0  # More sensitive to roughness
cfg.simulation_params['swing_velocity_sensitivity'] = 0.4  # More sensitive to speed
```

### Adjust Height Limits
```python
cfg.simulation_params['swing_height_min'] = 0.03  # Lower minimum
cfg.simulation_params['swing_height_max'] = 0.20  # Higher maximum
```

---

**Status:** ✅ Ready to use
**Testing:** ✅ All tests pass
**Backward Compatible:** ✅ Yes (disabled by default)
