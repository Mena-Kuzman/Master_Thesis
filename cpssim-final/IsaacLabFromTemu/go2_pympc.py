import sys

# Parse --headless BEFORE SimulationApp creation (SimulationApp must be
# the very first import, but it reads the headless flag at construction time).
_headless = "--headless" in sys.argv

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": _headless})

import argparse
import os

# Add paths for config import
# Ensure workspace root (one level up) is on sys.path so top-level packages are importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(repo_root, 'pympc'))

# Import config to check execution mode
from quadruped_pympc import config as cfg

from src.dog.dogWorld import DogWorld
from src.dog.dog_runner import DogRunner

# Initialize ROS if in ROS mode
if cfg.simulation_params.get('using_ros', False):
    try:
        import rclpy
        rclpy.init(args=None)
        print("[main] ROS initialized for controlled execution")
    except ImportError:
        print("[main] Warning: ROS mode selected but rclpy not available")
    except RuntimeError as e:
        print(f"[main] Warning: ROS already initialized or error: {e}")

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
parser.add_argument("--headless", action="store_true", default=False, help="Run without GUI")
parser.add_argument(
    "--rl-residual-checkpoint",
    type=str,
    default=cfg.simulation_params.get('rl_residual_checkpoint', None),
    help="Path to trained SAC checkpoint (*.pt) for step-height residual policy",
)
parser.add_argument(
    "--rl-residual-device",
    type=str,
    default=cfg.simulation_params.get('rl_residual_device', 'cpu'),
    choices=["cpu", "cuda"],
    help="Device for RL residual policy inference",
)
parser.add_argument(
    "--rl-residual-stochastic",
    action="store_true",
    default=bool(cfg.simulation_params.get('rl_residual_stochastic', False)),
    help="Use stochastic policy action instead of deterministic mean",
)
parser.add_argument(
    "--rl-residual-warmup-steps",
    type=int,
    default=int(cfg.simulation_params.get('rl_residual_warmup_steps', 120)),
    help="Physics steps with zero residual at startup",
)
parser.add_argument(
    "--rl-residual-ramp-steps",
    type=int,
    default=int(cfg.simulation_params.get('rl_residual_ramp_steps', 400)),
    help="Physics steps to ramp residual gate from 0 to 1",
)
args, unknown = parser.parse_known_args()


if __name__ == "__main__":

    world_cfg = DogWorld.Params()
    world = DogWorld(world_cfg)
    usainBolt = DogRunner(
        world,
        rl_residual_checkpoint=args.rl_residual_checkpoint,
        rl_residual_device=args.rl_residual_device,
        rl_residual_deterministic=(not args.rl_residual_stochastic),
        rl_residual_warmup_steps=args.rl_residual_warmup_steps,
        rl_residual_ramp_steps=args.rl_residual_ramp_steps,
    )
    usainBolt.run()    
    print("[main] Shutting down...")
    usainBolt.shutdown()
    
    # Cleanup ROS if it was initialized
    if cfg.simulation_params.get('using_ros', False):
        try:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
                print("[main] ROS shutdown complete")
        except:
            pass
    
