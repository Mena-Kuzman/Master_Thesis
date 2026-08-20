"""This file includes all of the configuration parameters for the MPC controllers
and of the internal simulations that can be launch from the folder /simulation.
"""
import numpy as np
from quadruped_pympc.helpers.quadruped_utils import GaitType
from quadruped_pympc.helpers.legs_attr import LegsAttr

# These are used both for a real experiment and a simulation -----------
# These are the only attributes needed per quadruped, the rest can be computed automatically ----------------------
robot = 'go2'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal1', 'hyqreal2', 'mini_cheetah'  # TODO: Load from robot_descriptions.py

try:
    from gym_quadruped.robot_cfgs import RobotConfig, get_robot_config
    robot_cfg: RobotConfig = get_robot_config(robot_name=robot)
    robot_leg_joints = robot_cfg.leg_joints
    robot_feet_geom_names = robot_cfg.feet_geom_names
    qpos0_js = robot_cfg.qpos0_js
    hip_height = robot_cfg.hip_height   #0.28 m
except Exception:
    class RobotConfig:
        def __init__(self, leg_joints, feet_geom_names, qpos0_js, hip_height):
            self.leg_joints = leg_joints
            self.feet_geom_names = feet_geom_names
            self.qpos0_js = qpos0_js
            self.hip_height = hip_height

    def get_robot_config(robot_name: str):
        if robot_name == 'go2':
            return RobotConfig(
                leg_joints=LegsAttr(
                    FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'],
                    FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'],
                    RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'],
                    RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'],
                ),
                feet_geom_names=LegsAttr(FL='FL_foot', FR='FR_foot', RL='RL_foot', RR='RR_foot'),
                qpos0_js=np.array([0.0, 0.7, -1.4] * 4, dtype=np.float64),
                hip_height=0.28,
            )
        raise ValueError(f"Unsupported robot without gym_quadruped installed: {robot_name}")

    robot_cfg: RobotConfig = get_robot_config(robot_name=robot)
    robot_leg_joints = robot_cfg.leg_joints
    robot_feet_geom_names = robot_cfg.feet_geom_names
    qpos0_js = robot_cfg.qpos0_js
    hip_height = robot_cfg.hip_height

# ----------------------------------------------------------------------------------------------------------------
if (robot == 'go1'):
    mass = 12.019
    inertia = np.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                        [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                        [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])

elif (robot == 'go2'):
    mass = 15.019
    """inertia = np.array([[0.02448, 1.21660000e-04, 0.0014849],
                        [1.21660000e-04, 0.098077, -3.12000000e-05],
                        [0.0014849, -3.12000000e-05, 0.107]])"""
    inertia = np.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                        [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                        [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])

elif (robot == 'aliengo'):
    mass = 24.637
    inertia = np.array([[0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
                        [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
                        [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808]])

elif (robot == 'b2'):
    mass = 83.49
    inertia = np.array([[0.2310941359705289, -0.0014987128245817424, -0.021400468992761768],
                        [-0.0014987128245817424, 1.4485084687476608, 0.0004641447134275615],
                        [-0.021400468992761768, 0.0004641447134275615, 1.503217877350808]])


elif (robot == 'hyqreal1'):
    mass = 108.40
    inertia = np.array([[4.55031444e+00, 2.75249434e-03, -5.11957307e-01],
                        [2.75249434e-03, 2.02411774e+01, -7.38560592e-04],
                        [-5.11957307e-01, -7.38560592e-04, 2.14269772e+01]])
    
elif (robot == 'hyqreal2'):
    mass = 126.69
    inertia = np.array([[4.55031444e+00, 2.75249434e-03, -5.11957307e-01],
                        [2.75249434e-03, 2.02411774e+01, -7.38560592e-04],
                        [-5.11957307e-01, -7.38560592e-04, 2.14269772e+01]])
    
elif (robot == 'mini_cheetah'):
    mass = 12.5
    inertia = np.array([[1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
                        [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
                        [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]])


gravity_constant = 9.81 # Exposed in case of different gravity conditions
# ----------------------------------------------------------------------------------------------------------------

mpc_params = {
    # 'nominal' optimized directly the GRF
    # 'input_rates' optimizes the delta GRF
    # 'sampling' is a gpu-based mpc that samples the GRF
    # 'collaborative' optimized directly the GRF and has a passive arm model inside
    # 'lyapunov' optimized directly the GRF and has a Lyapunov-based stability constraint
    # 'kinodynamic' sbrd with joints - experimental
    'type':                                    'nominal',  # 'nominal', 'input_rates', 'sampling', 'collaborative', 'lyapunov', 'kinodynamic'

    # print the mpc info
    'verbose':                                 False,

    # horizon is the number of timesteps in the future that the mpc will optimize
    # dt is the discretization time used in the mpc
    'horizon':                                 12,
    'dt':                                      0.02,

    # GRF limits for each single leg
    "grf_max":                                 2 * mass * gravity_constant,
    "grf_min":                                 0,
    'mu':                                      0.5,

    # this is used to have a smaller dt near the start of the horizon
    'use_nonuniform_discretization':           True,
    'horizon_fine_grained':                    4,
    'dt_fine_grained':                         0.01,

    # if this is true, we optimize the step frequency as well
    # for the sampling controller, this is done in the rollout
    # for the gradient-based controller, this is done with a batched version of the ocp
    # useless in case of adaptive gait sequencing, because the period is calculated from the stride length and velocity
    'optimize_step_freq':                      False,
    'step_freq_available':                     [1.4, 2.0, 2.4],

    # ----- START properties only for the gradient-based mpc -----

    # this is used if you want to manually warm start the mpc
    'use_warm_start':                          False,

    # this enables integrators for height, linear velocities, roll and pitch
    'use_integrators':                         False,
    'alpha_integrator':                        0.1,
    'integrator_cap':                          [0.5, 0.2, 0.2, 0.0, 0.0, 1.0],

    # if this is off, the mpc will not optimize the footholds and will
    # use only the ones provided in the reference
    'use_foothold_optimization':               True,

    # this is set to false automatically is use_foothold_optimization is false
    # because in that case we cannot chose the footholds and foothold
    # constraints do not any make sense
    'use_foothold_constraints':                True,

    # works with all the mpc types except 'sampling'. In sim does not do much for now,
    # but in real it minizimes the delay between the mpc control and the state
    'use_RTI':                                 False,
    # If RTI is used, we can set the advance RTI-step! (Standard is the simpler RTI)
    # See https://arxiv.org/pdf/2403.07101.pdf
    'as_rti_type':                             "Standard",  # "AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D", "Standard"
    'as_rti_iter':                             1,  # > 0, the higher the better, but slower computation!

    # This will force to use DDP instead of SQP, based on https://arxiv.org/abs/2403.10115.
    # Note that RTI is not compatible with DDP, and no state costraints for now are considered
    'use_DDP':                                 False,

    # this is used only in the case 'use_RTI' is false in a single mpc feedback loop.
    # More is better, but slower computation!
    'num_qp_iterations':                       20,

    # Maximum iterations for the internal QP solver (HPIPM).
    # Useful to reduce persistent ACADOS_MAXITER (status=2) events.
    'qp_solver_iter_max':                      200,

    # this is used to speeding up or robustify acados' solver (hpipm).
    'solver_mode':                             'balance',  # balance, robust, speed, crazy_speed


    # these is used only for the case 'input_rates', using as GRF not the actual state
    # of the robot of the predicted one. Can be activated to compensate
    # for the delay in the control loop on the real robot
    'use_input_prediction':                    False,

    # ONLY ONE CAN BE TRUE AT A TIME (only gradient)
    'use_static_stability':                    False,
    'use_zmp_stability':                       False,
    'trot_stability_margin':                   0.04,
    'pace_stability_margin':                   0.1,
    'crawl_stability_margin':                  0.04,  # in general, 0.02 is a good value

    # this is used to compensate for the external wrenches
    # you should provide explicitly this value in compute_control
    'external_wrenches_compensation':          True,
    'external_wrenches_compensation_num_step': 15,

    # this is used only in the case of collaborative mpc, to
    # compensate for the external wrench in the prediction (only collaborative)
    'passive_arm_compensation':                True,


    # Gain for Lyapunov-based MPC
    'K_z1': np.array([1, 1, 10]),
    'K_z2': np.array([1, 4, 10]),
    'residual_dynamics_upper_bound': 30,
    'use_residual_dynamics_decay': False,

    # ----- END properties for the gradient-based mpc -----


    # ----- START properties only for the sampling-based mpc -----

    # this is used only in the case 'sampling'.
    'sampling_method':                         'random_sampling',  # 'random_sampling', 'mppi', 'cem_mppi'
    'control_parametrization':                 'cubic_spline', # 'cubic_spline', 'linear_spline', 'zero_order'
    'num_splines':                             2,  # number of splines to use for the control parametrization
    'num_parallel_computations':               10000,  # More is better, but slower computation!
    'num_sampling_iterations':                 1,  # More is better, but slower computation!
    'device':                                  'gpu',  # 'gpu', 'cpu'
    # convariances for the sampling methods
    'sigma_cem_mppi':                          3,
    'sigma_mppi':                              3,
    'sigma_random_sampling':                   [0.2, 3, 10],
    'shift_solution':                          False,

    # ----- END properties for the sampling-based mpc -----
    }
# -----------------------------------------------------------------------

simulation_params = {
    # Execution mode: True = ROS-based control, False = direct Isaac Sim control
    'using_ros':                   False,

    'swing_generator':             'scipy',  # 'scipy', 'explicit'
    'swing_position_gain_fb':      500,
    'swing_velocity_gain_fb':      10,
    'impedence_joint_position_gain':  10.0,
    'impedence_joint_velocity_gain':  2.0,

    'step_height':                 0.2 * hip_height,  

    # Obstacle-adaptive step height parameters (for blind robot stair/obstacle climbing)
    'obstacle_retry_enabled': False,             # Deployment default: let RL residual own step-height adaptation
    'obstacle_retry_max_boost': 0.6 * hip_height,  # Maximum additional step height boost for obstacle retry
    'obstacle_retry_boost_increment': 0.2 * hip_height,  # Boost increment per failed retry attempt
    'obstacle_retry_max_attempts': 5,            # Maximum retry attempts before giving up on obstacle
    'obstacle_retry_hold_cycles': 10,              # Cycles to maintain boosted step height before reset

    # Obstacle-triggered speed reduction and recovery
    'obstacle_slowdown_enabled': False,                       # DISABLED: Reduce commanded speed after obstacle hit
    'obstacle_slowdown_min_activation_speed': 0.4,            # Minimum speed (m/s) to trigger slowdown on obstacle hit
    'obstacle_slowdown_speed_threshold': 0.8,                 # Speed threshold for switching slowdown behavior (m/s)
    'obstacle_slowdown_low_speed_percentage': 0.4,            # Slowdown percentage for speeds <= threshold (40%)
    'obstacle_slowdown_high_speed_percentage': 0.5,           # Slowdown percentage for speeds > threshold (50%), this is the working value.
    'obstacle_slowdown_ramp_rate': 0.2,                       # Speed recovery rate (m/s per second)
    'obstacle_slowdown_heightmap_threshold': 0.03,            # Height delta (m) above terrain to trigger slowdown in heightmap mode
    'obstacle_slowdown_terrain_pitch_tolerance': 0.1,        # Tolerance for terrain pitch (rad) to detect stability for ramp rate activation
    'obstacle_slowdown_stable_cycles': 5,                    # Cycles of stable terrain required before ramp resumes
    'obstacle_slowdown_ramp_enabled': True,                  # Enable ramp-up to previous speed when terrain pitch is constant

    'terrain_pitch_speed_limit_enabled': False,               # Optional steep-slope speed limit (keep disabled unless needed)
    'terrain_pitch_speed_limit_threshold': 0.2,               # Activate limit above |pitch| (rad)
    'terrain_pitch_speed_limit_factor': 0.5,                  # Multiply forward command when limit is active
    'terrain_pitch_vertical_scale': 2.0,                      # Scale vertical component when limit is active

    'debug_contact_logging': False,            # Log current_contact vs early_stance
    'debug_contact_logging_stride': 100,        # Log every N control cycles
    'debug_early_stance_logging': False,        # Log early_stance detection details (swing_time, disp, error_ratio, contact)


    # Visual Foothold adapatation
    "visual_foothold_adaptation":  'height', #'blind', 'height', 'vfa'
    
    # Heightmap parameters (for visual foothold adaptation)
    'heightmap_resolution': 0.06,               # Grid cell size in meters (default: 4cm per cell)
    'heightmap_num_rows': 7,                     # Number of rows in heightmap grid
    'heightmap_num_cols': 7,                     # Number of columns in heightmap grid
    'heightmap_raycast_max_dist': 1.5,           # Raycast max distance (meters) for HeightMapIsaac
    'heightmap_height_bias': 0.0,                # Added to sampled terrain z before touchdown reference
    'heightmap_self_hit_exclusion_radius': 0.12,  # Exclude raycast hits within this radius (m) of robot bodies
    'heightmap_self_hit_z_tolerance': 0.015,      # Reject as self-hit only when z is also very close to body origin
    'heightmap_disable_at_zero_cmd': True,       # If true, bypass visual adaptation whenever commanded lin/yaw velocity is near zero
    'heightmap_zero_cmd_threshold': 0.02,        # m/s (+ rad/s yaw term) below which command is treated as zero

    # Total prediction range = num_rows × resolution (default: 7 × 0.04m = 0.28m ahead)

    # RL residual step-height assist (MPC remains primary controller)
    'rl_step_height_enabled': False,               # Global default. RL training env overrides this to True at runtime.
    'rl_residual_checkpoint': 'checkpoints/sac_step_height_residual_seed1_1774263860_final.pt',  # Default SAC residual checkpoint used by Isaac runtime scripts
    'rl_residual_device': 'cuda',                  # Inference device for residual actor: 'cpu' or 'cuda'
    'rl_residual_stochastic': False,              # False => deterministic mean action, True => sample stochastic action
    'rl_residual_warmup_steps': 50,              # Runtime startup warmup steps with zero residual
    'rl_residual_ramp_steps': 200,                # Runtime residual gate ramp duration after warmup
    'rl_residual_runtime_extra_gates_enabled': False,  # Apply runtime-only speed/failure residual gates (kept False to match training semantics)
    'rl_residual_speed_blend_start': 0.2,        # Full RL residual at/under this planar command speed [m/s]
    'rl_residual_speed_blend_zero': 1.5,         # RL residual fully disabled at/above this planar command speed [m/s]
    'rl_step_height_residual_max_abs': 0.6 * hip_height,       # Clamp |residual| in meters
    'rl_step_height_residual_smoothing': 0.25,     # EMA smoothing factor [0..1] for RL residual
    'rl_step_height_min': 0.16* hip_height,       # Minimum effective swing step height (scaled by hip_height)
    'rl_step_height_max': 0.9 * hip_height,       # Maximum effective swing step height (scaled by hip_height)
    'rl_residual_disable_on_mpc_failure': True,   # Force RL residual gate to zero when MPC reports hard failure

    # this is the integration time used in the simulator
    'dt':                          0.002,

    'gait':                        'trot',  # 'trot', 'pace', 'crawl', 'bound', 'full_stance', 'galloping'
    'gait_params':                 {'trot': {'step_freq': 1.4, 'duty_factor': 0.65, 'type': GaitType.TROT.value},
                                    'crawl': {'step_freq': 0.5, 'duty_factor': 0.8, 'type': GaitType.BACKDIAGONALCRAWL.value},
                                    'pace': {'step_freq': 1.4, 'duty_factor': 0.7, 'type': GaitType.PACE.value},
                                    'bound': {'step_freq': 1.8, 'duty_factor': 0.65, 'type': GaitType.BOUNDING.value},
                                    'full_stance': {'step_freq': 2, 'duty_factor': 0.65, 'type': GaitType.FULL_STANCE.value},
                                    'galloping': {'stq': 2.0, 'duty_factor': 0.3, 'type': GaitType.GALLOPING.value, 'step_height': 0.15},
                                   },

    # This is used to activate or deactivate the reflexes upon contact detection
    'reflex_trigger_mode':       False, # 'tracking', 'geom_contact', False
    'prelearning_reflex_trigger_mode': 'tracking',  # Reflex trigger mode used before learning_starts_steps in RL env
    'reflex_max_step_height':    0.6 * hip_height,  # this is the maximum step height that the robot can do if reflexes are enabled
    'reflex_next_steps_height_enhancement': False,
    'prelearning_reflex_next_steps_height_enhancement': False,  # Next-step reflex boost used before learning starts
    'reflex_contact_force_threshold': 1.5,   # N, used when geom_contact receives raw Isaac foot forces
    'reflex_contact_binary_threshold': 0.5,  # threshold when geom_contact receives binary contact states

    # Isaac foot contact processing (DogRunner hysteresis/debounce)
    'isaac_contact_on_threshold': 2.5,       # N
    'isaac_contact_off_threshold': 1.0,      # N
    'isaac_contact_debounce_steps': 1,
    'velocity_modulator': False,

    # velocity mode: human will give you the possibility to use the keyboard, the other are
    # forward only random linear-velocity, random will give you random linear-velocity and yaw-velocity
    'mode':                        'human',  # 'human', 'forward', 'random'
    'ref_z':                       0.9 *hip_height,


    # the MPC will be called every 1/(mpc_frequency*dt) timesteps
    # this helps to evaluate more realistically the performance of the controller
    'mpc_frequency':               100,  # in Hz

    'use_inertia_recomputation':   True,

    'scene':                       'flat',  # flat, random_boxes, random_pyramids, perlin

    # Adaptive gait sequencing (DFKI AdaptiveGait-style)
    'adaptive_gait_enabled':       True,
    'adaptive_swing_time':          0.2,  # Nominal swing time used as the center of the RL-controlled range
    'adaptive_swing_time_min':      0.15, # Lower bound for RL-controlled swing time (s)
    'adaptive_swing_time_max':      0.22, # Upper bound for RL-controlled swing time (s)
    'adaptive_swing_time_smoothing': 0.25,
    'adaptive_min_velocity':        0.02,
    'adaptive_min_period':          0.3,  # Minimum period to ensure valid duty factor (period ≥ swing_time / (1 - min_duty_factor))
    'adaptive_max_period':          0.60,  # Maximum period at low velocities to reduce stance time (caps duty_factor to ~50%)
    'adaptive_max_stride_length':   0.75,
    'adaptive_zero_velocity_threshold': 0.1,
    'adaptive_standing_foot_position_threshold': 0.07,
    'adaptive_disturbance_correction': 1.3,
    'adaptive_activation_speed':    0.0,  # Only activate adaptive gait above this speed

    }