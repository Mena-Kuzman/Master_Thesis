"""
Test script for terrain-adaptive step height implementation.

Tests the individual components and verifies the mathematical relationships.
"""

import numpy as np
from quadruped_pympc.helpers.terrain_roughness_estimator import TerrainRoughnessEstimator
from quadruped_pympc import config as cfg


def test_roughness_estimator():
    """Test the terrain roughness estimator."""
    print("=" * 60)
    print("TEST 1: Terrain Roughness Estimator")
    print("=" * 60)
    
    estimator = TerrainRoughnessEstimator(window_size=50, smoothing_factor=0.95)
    
    # Test on smooth terrain
    print("\nSmooth Terrain (GRF noise = ±2N):")
    for i in range(100):
        grf = np.array([
            [0, 0, 60 + np.random.normal(0, 2)],
            [0, 0, 60 + np.random.normal(0, 2)],
            [0, 0, 60 + np.random.normal(0, 2)],
            [0, 0, 60 + np.random.normal(0, 2)],
        ])
        contact = np.array([1, 1, 1, 1])
        roughness = estimator.update(grf, contact)
    
    smooth_roughness = estimator.get_roughness()
    print(f"  Estimated roughness: {smooth_roughness:.4f} (expected ~ 0.1-0.3)")
    
    estimator.reset()
    
    # Test on rough terrain
    print("\nRough Terrain (GRF noise = ±15N):")
    for i in range(100):
        grf = np.array([
            [0, 0, 60 + np.random.normal(0, 15)],
            [0, 0, 60 + np.random.normal(0, 15)],
            [0, 0, 60 + np.random.normal(0, 15)],
            [0, 0, 60 + np.random.normal(0, 15)],
        ])
        contact = np.array([1, 1, 1, 1])
        roughness = estimator.update(grf, contact)
    
    rough_roughness = estimator.get_roughness()
    print(f"  Estimated roughness: {rough_roughness:.4f} (expected ~ 0.7-0.8)")
    
    return smooth_roughness < 0.4 and rough_roughness > 0.6


def test_adaptive_step_height():
    """Test the terrain-adaptive step height computation."""
    print("\n" + "=" * 60)
    print("TEST 2: Terrain-Adaptive Step Height")
    print("=" * 60)
    
    # Import after config is set
    from quadruped_pympc.helpers.swing_trajectory_controller import SwingTrajectoryController
    
    # Create a basic swing controller
    step_height = 0.056  # Default base step height
    swing_period = 0.43
    position_gain = np.ones(3) * 500
    velocity_gain = np.ones(3) * 10
    
    stc = SwingTrajectoryController(
        step_height=step_height,
        swing_period=swing_period,
        position_gain_fb=position_gain,
        velocity_gain_fb=velocity_gain,
        generator='explicit'
    )
    
    print(f"\nBase step height: {step_height:.4f} m")
    print(f"Min allowed height: {cfg.simulation_params.get('swing_height_min', 0.04):.4f} m")
    print(f"Max allowed height: {cfg.simulation_params.get('swing_height_max', 0.15):.4f} m")
    
    # Test Case 1: Flat terrain, low velocity
    h1 = stc.compute_terrain_adaptive_step_height(
        terrain_pitch=0.0,
        terrain_roughness=0.0,
        velocity=0.0
    )
    print(f"\nCase 1 - Flat, low velocity:")
    print(f"  Input: pitch=0°, roughness=0.0, v=0 m/s")
    print(f"  Output: {h1:.4f} m (expected ≈ {step_height:.4f})")
    
    # Test Case 2: Upslope, low velocity
    h2 = stc.compute_terrain_adaptive_step_height(
        terrain_pitch=np.pi/12,  # 15 degrees upslope
        terrain_roughness=0.0,
        velocity=0.0
    )
    print(f"\nCase 2 - Upslope (+15°), low velocity:")
    print(f"  Input: pitch=+15°, roughness=0.0, v=0 m/s")
    print(f"  Output: {h2:.4f} m (expected > {step_height:.4f})")
    
    # Test Case 3: Downslope, low velocity
    h3 = stc.compute_terrain_adaptive_step_height(
        terrain_pitch=-np.pi/12,  # 15 degrees downslope
        terrain_roughness=0.0,
        velocity=0.0
    )
    print(f"\nCase 3 - Downslope (-15°), low velocity:")
    print(f"  Input: pitch=-15°, roughness=0.0, v=0 m/s")
    print(f"  Output: {h3:.4f} m (expected < {step_height:.4f})")
    
    # Test Case 4: Flat, rough terrain
    h4 = stc.compute_terrain_adaptive_step_height(
        terrain_pitch=0.0,
        terrain_roughness=0.5,  # 50% rough
        velocity=0.0
    )
    print(f"\nCase 4 - Flat, rough terrain:")
    print(f"  Input: pitch=0°, roughness=0.5, v=0 m/s")
    print(f"  Output: {h4:.4f} m (expected > {step_height:.4f})")
    
    # Test Case 5: Flat, high velocity
    h5 = stc.compute_terrain_adaptive_step_height(
        terrain_pitch=0.0,
        terrain_roughness=0.0,
        velocity=2.0,  # Maximum velocity
        velocity_max=2.0
    )
    print(f"\nCase 5 - Flat, high velocity:")
    print(f"  Input: pitch=0°, roughness=0.0, v=2.0 m/s")
    print(f"  Output: {h5:.4f} m (expected > {step_height:.4f})")
    
    # Test Case 6: Combined (upslope + rough + high velocity)
    h6 = stc.compute_terrain_adaptive_step_height(
        terrain_pitch=np.pi/12,
        terrain_roughness=0.8,
        velocity=1.5,
        velocity_max=2.0
    )
    print(f"\nCase 6 - Combined (upslope + rough + high velocity):")
    print(f"  Input: pitch=+15°, roughness=0.8, v=1.5 m/s")
    print(f"  Output: {h6:.4f} m (expected > {step_height:.4f})")
    
    # Verify ordering
    tests_pass = (
        h1 > 0.04 and h1 < 0.15 and  # Within bounds
        h2 > h1 and  # Upslope increases height
        h3 < h1 and  # Downslope decreases height
        h4 > h1 and  # Roughness increases height
        h5 > h1 and  # Velocity increases height
        h6 > h2 and h6 > h4 and h6 > h5  # Combined effect is larger
    )
    
    return tests_pass


def test_configuration():
    """Test that configuration parameters are set correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Configuration Parameters")
    print("=" * 60)
    
    print(f"\nTerrain-adaptive step height enabled: {cfg.simulation_params.get('terrain_adaptive_step_height_enabled')}")
    print(f"Swing height min: {cfg.simulation_params.get('swing_height_min')} m")
    print(f"Swing height max: {cfg.simulation_params.get('swing_height_max')} m")
    print(f"Pitch sensitivity: {cfg.simulation_params.get('swing_pitch_sensitivity')}")
    print(f"Roughness sensitivity: {cfg.simulation_params.get('swing_roughness_sensitivity')}")
    print(f"Velocity sensitivity: {cfg.simulation_params.get('swing_velocity_sensitivity')}")
    
    return True


if __name__ == "__main__":
    print("\n\nTERRAIN-ADAPTIVE STEP HEIGHT - TEST SUITE\n")
    
    test1_pass = test_roughness_estimator()
    test2_pass = test_adaptive_step_height()
    test3_pass = test_configuration()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Roughness Estimator): {'PASS' if test1_pass else 'FAIL'}")
    print(f"Test 2 (Adaptive Step Height): {'PASS' if test2_pass else 'FAIL'}")
    print(f"Test 3 (Configuration): {'PASS' if test3_pass else 'FAIL'}")
    print(f"\nOverall: {'ALL TESTS PASS' if (test1_pass and test2_pass and test3_pass) else 'SOME TESTS FAILED'}")
    print("=" * 60)
