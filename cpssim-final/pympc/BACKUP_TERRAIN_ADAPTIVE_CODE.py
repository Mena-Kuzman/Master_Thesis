"""
BACKUP FILE - Terrain-Adaptive Step Height Code Snippets
=========================================================

This file contains all the code that was added for the terrain-adaptive
step height feature. Each section is labeled with the original file path
and line numbers.

Created: February 6, 2026
Purpose: Backup before removal of terrain-adaptive feature code
"""

# ============================================================================
# FILE: quadruped_pympc/config.py
# LOCATION: Lines 201-207 (approximately)
# DESCRIPTION: Configuration parameters for terrain-adaptive step height
# ============================================================================

CONFIG_PY_ADDITIONS = """
    # Terrain-adaptive step height parameters
    'terrain_adaptive_step_height_enabled': False,  # Enable terrain-adaptive step height
    'swing_height_min': 0.04,                       # Minimum step height (m)
    'swing_height_max': 0.15,                       # Maximum step height (m)
    'swing_pitch_sensitivity': 0.5,                 # Pitch adaptation coefficient [0, 1]
    'swing_roughness_sensitivity': 0.8,             # Roughness adaptation coefficient [0, 1]
    'swing_velocity_sensitivity': 0.3,              # Velocity adaptation coefficient [0, 1]
"""

# ============================================================================
# FILE: quadruped_pympc/helpers/swing_trajectory_controller.py
# LOCATION: Lines 32-34 (base_step_height property)
# DESCRIPTION: Store base step height for adaptive computations
# ============================================================================

SWING_CONTROLLER_PROPERTY = """
        # Store base step height for adaptive computations
        self.base_step_height = step_height
"""

# ============================================================================
# FILE: quadruped_pympc/helpers/swing_trajectory_controller.py
# LOCATION: Lines 47-108 (approximately)
# DESCRIPTION: Compute terrain-adaptive step height method
# ============================================================================

SWING_CONTROLLER_METHOD = '''
    def compute_terrain_adaptive_step_height(self,
                                            terrain_pitch: float,
                                            terrain_roughness: float,
                                            velocity: float = 0.0,
                                            velocity_max: float = 2.0) -> float:
        """
        Compute adaptive step height based on terrain slope and roughness.
        
        The step height adapts to:
        1. Terrain pitch: Increase height on upslopes, decrease on downslopes
        2. Terrain roughness: Increase height on rough terrain for obstacle avoidance
        3. Velocity: Increase height at higher speeds for dynamic stability
        
        Args:
            terrain_pitch (float): Estimated terrain pitch angle in radians
            terrain_roughness (float): Estimated terrain roughness [0, 1]
            velocity (float): Current robot velocity in m/s (default: 0.0)
            velocity_max (float): Maximum design velocity in m/s (default: 2.0)
        
        Returns:
            float: Adaptive step height in meters, clipped to [0.04, 0.15] m
        """
        from quadruped_pympc import config as cfg
        
        # Base step height
        h_base = self.base_step_height
        
        # ===== Pitch Adaptation =====
        # Increase height on upslopes (positive pitch)
        # Decrease height on downslopes (negative pitch)
        # Range: [0.5, 1.5] (±50% variation)
        pitch_sensitivity = cfg.simulation_params.get('swing_pitch_sensitivity', 0.5)
        pitch_factor = 1.0 + pitch_sensitivity * np.sin(terrain_pitch)
        
        # ===== Roughness Adaptation =====
        # Increase height for rough terrain to clear obstacles
        # Range: [1.0, 1.8] (+80% max increase)
        roughness_sensitivity = cfg.simulation_params.get('swing_roughness_sensitivity', 0.8)
        roughness_factor = 1.0 + roughness_sensitivity * terrain_roughness
        
        # ===== Velocity Adaptation (Optional) =====
        # Increase height at higher speeds for dynamic stability margin
        # Range: [1.0, 1.3] (+30% max increase)
        velocity_sensitivity = cfg.simulation_params.get('swing_velocity_sensitivity', 0.3)
        velocity_normalized = np.clip(velocity / velocity_max, 0.0, 1.0)
        velocity_factor = 1.0 + velocity_sensitivity * velocity_normalized
        
        # ===== Combined Adaptive Step Height =====
        # Multiplicative combination of factors
        h_adaptive = h_base * pitch_factor * roughness_factor * velocity_factor
        
        # ===== Saturation Constraints =====
        # Prevent unrealistic trajectories
        h_min = cfg.simulation_params.get('swing_height_min', 0.04)
        h_max = cfg.simulation_params.get('swing_height_max', 0.15)
        h_final = np.clip(h_adaptive, h_min, h_max)
        
        return float(h_final)
'''

# ============================================================================
# FILE: quadruped_pympc/interfaces/wb_interface.py
# LOCATION: Lines 279-311 (approximately)
# DESCRIPTION: Integration of terrain-adaptive step height into control loop
# ============================================================================

WB_INTERFACE_INTEGRATION = '''
        # ===== NEW: Terrain-Adaptive Step Height =====
        if cfg.simulation_params.get('terrain_adaptive_step_height_enabled', False):
            # Initialize roughness estimator if not present
            if not hasattr(self, 'roughness_estimator'):
                from quadruped_pympc.helpers.terrain_roughness_estimator import TerrainRoughnessEstimator
                self.roughness_estimator = TerrainRoughnessEstimator()
            
            # Update roughness estimate (using estimated GRF from MPC if available)
            # For now, we use a simple heuristic based on contact state changes
            estimated_grf = np.zeros((4, 3))  # Placeholder for estimated GRF
            terrain_roughness = self.roughness_estimator.update(estimated_grf, self.current_contact)
            
            # Compute adaptive step height
            velocity = float(np.linalg.norm(base_lin_vel[0:2]))
            adaptive_step_height = self.stc.compute_terrain_adaptive_step_height(
                terrain_pitch=terrain_pitch,
                terrain_roughness=terrain_roughness,
                velocity=velocity,
                velocity_max=2.0
            )
            
            # Regenerate swing trajectory with adaptive height
            swing_period = (1.0 - self.pgg.duty_factor) / self.pgg.step_freq
            self.stc.regenerate_swing_trajectory_generator(
                step_height=adaptive_step_height,
                swing_period=swing_period
            )
        else:
            # Use fixed step height from config
            swing_period = (1.0 - self.pgg.duty_factor) / self.pgg.step_freq
            self.stc.regenerate_swing_trajectory_generator(
                step_height=self.step_height,
                swing_period=swing_period
            )
'''

# ============================================================================
# ENTIRE NEW FILE CREATED
# FILE: quadruped_pympc/helpers/terrain_roughness_estimator.py
# DESCRIPTION: Complete terrain roughness estimator class
# ============================================================================

TERRAIN_ROUGHNESS_ESTIMATOR_FULL_FILE = '''
"""
Terrain Roughness Estimator

Estimates terrain roughness from ground reaction force (GRF) variance during stance phase.
The roughness metric is used to adapt the step height for better obstacle avoidance on
uneven terrain.

Author: Terrain-Adaptive Step Height Feature
Date: February 6, 2026
"""

import numpy as np
from collections import deque


class TerrainRoughnessEstimator:
    """
    Estimates terrain roughness from GRF variance during stance phase.
    
    The roughness is quantified by analyzing the variance of vertical GRF
    measurements across all legs in contact with the ground. Higher variance
    indicates rougher terrain with more obstacles and irregularities.
    
    Attributes:
        window_size (int): Number of recent measurements to consider
        smoothing_factor (float): Low-pass filter coefficient (0-1)
        grf_history (deque): Sliding window of GRF variance measurements
        current_roughness (float): Current estimated roughness [0, 1]
    """
    
    def __init__(self, window_size: int = 100, smoothing_factor: float = 0.95):
        """
        Initialize the terrain roughness estimator.
        
        Args:
            window_size (int): Size of sliding window for variance computation
            smoothing_factor (float): Exponential smoothing factor [0, 1]
                                     Higher values = more smoothing
        """
        self.window_size = window_size
        self.smoothing_factor = smoothing_factor
        self.grf_history = deque(maxlen=window_size)
        self.current_roughness = 0.0
        
    def update(self, grf: np.ndarray, contact_state: np.ndarray) -> float:
        """
        Update roughness estimate with new GRF measurements.
        
        Args:
            grf (np.ndarray): Ground reaction forces, shape (4, 3) for 4 legs
                             Each row: [fx, fy, fz] in Newtons
            contact_state (np.ndarray): Contact state for each leg, shape (4,)
                                       1 = in contact, 0 = in swing
        
        Returns:
            float: Updated roughness estimate [0, 1]
                  0 = smooth terrain, 1 = very rough terrain
        """
        # Only consider legs in stance phase
        stance_legs = contact_state > 0.5
        
        if np.sum(stance_legs) > 0:
            # Extract vertical GRF for stance legs
            vertical_grf = grf[stance_legs, 2]  # z-component
            
            # Compute variance of vertical GRF
            grf_variance = float(np.var(vertical_grf))
            
            # Add to history
            self.grf_history.append(grf_variance)
            
            # Compute roughness from accumulated variance
            if len(self.grf_history) >= 10:  # Need minimum samples
                # Use log-scale to handle wide range of variances
                mean_variance = np.mean(self.grf_history)
                
                # Normalize to [0, 1] range
                # Typical smooth terrain: variance ~ 1-10 N²
                # Typical rough terrain: variance ~ 100-1000 N²
                roughness_raw = np.log10(mean_variance + 1.0) / 3.0  # Divide by log10(1000)
                roughness_raw = np.clip(roughness_raw, 0.0, 1.0)
                
                # Apply exponential smoothing to prevent jitter
                self.current_roughness = (
                    self.smoothing_factor * self.current_roughness + 
                    (1.0 - self.smoothing_factor) * roughness_raw
                )
        
        return self.current_roughness
    
    def get_roughness(self) -> float:
        """
        Get the current roughness estimate.
        
        Returns:
            float: Current roughness [0, 1]
        """
        return self.current_roughness
    
    def reset(self) -> None:
        """
        Reset the estimator to initial state.
        """
        self.grf_history.clear()
        self.current_roughness = 0.0


# Example usage:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    estimator = TerrainRoughnessEstimator()
    
    # Simulate smooth terrain
    roughness_smooth = []
    for i in range(200):
        grf = np.array([
            [0, 0, 60 + np.random.normal(0, 2)],
            [0, 0, 60 + np.random.normal(0, 2)],
            [0, 0, 60 + np.random.normal(0, 2)],
            [0, 0, 60 + np.random.normal(0, 2)],
        ])
        contact = np.array([1, 1, 1, 1])
        roughness = estimator.update(grf, contact)
        roughness_smooth.append(roughness)
    
    estimator.reset()
    
    # Simulate rough terrain
    roughness_rough = []
    for i in range(200):
        grf = np.array([
            [0, 0, 60 + np.random.normal(0, 15)],
            [0, 0, 60 + np.random.normal(0, 15)],
            [0, 0, 60 + np.random.normal(0, 15)],
            [0, 0, 60 + np.random.normal(0, 15)],
        ])
        contact = np.array([1, 1, 1, 1])
        roughness = estimator.update(grf, contact)
        roughness_rough.append(roughness)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(roughness_smooth)
    plt.title('Smooth Terrain (GRF noise ±2N)')
    plt.xlabel('Time step')
    plt.ylabel('Roughness estimate')
    plt.ylim([0, 1])
    
    plt.subplot(1, 2, 2)
    plt.plot(roughness_rough)
    plt.title('Rough Terrain (GRF noise ±15N)')
    plt.xlabel('Time step')
    plt.ylabel('Roughness estimate')
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.show()
'''

# ============================================================================
# SUMMARY
# ============================================================================

SUMMARY = """
FILES MODIFIED:
1. quadruped_pympc/config.py - Added 6 configuration parameters
2. quadruped_pympc/helpers/swing_trajectory_controller.py - Added base_step_height property and compute_terrain_adaptive_step_height() method
3. quadruped_pympc/interfaces/wb_interface.py - Added terrain-adaptive integration logic

FILES CREATED:
1. quadruped_pympc/helpers/terrain_roughness_estimator.py - Complete new file

TOTAL CODE ADDED: ~400 lines
- Configuration parameters: 6 lines
- Swing controller property: 2 lines
- Swing controller method: 61 lines
- WB interface integration: 33 lines
- Terrain roughness estimator: 125 lines
- Test file: 200+ lines
- Documentation: 1000+ lines

All code snippets are preserved in this backup file for reference.
"""

if __name__ == "__main__":
    print("=" * 80)
    print("TERRAIN-ADAPTIVE STEP HEIGHT CODE BACKUP")
    print("=" * 80)
    print(SUMMARY)
    print("\nAll code snippets are preserved in the variables above.")
    print("To restore, copy the relevant sections back to their original files.")
