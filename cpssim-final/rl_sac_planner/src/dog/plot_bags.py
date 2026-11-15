#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

try:
    from dls2_msgs.msg import BaseStateMsg, ControlSignalMsg, TrajectoryGeneratorMsg, BlindStateMsg
except ImportError:
    print("Error: Could not import dls2_msgs. Source your workspace.", file=sys.stderr)
    sys.exit(1)

TOPIC_MAP = {
    '/dls2/base_state': 'dls2_msgs/msg/BaseStateMsg',
    '/dls2/blind_state': 'dls2_msgs/msg/BlindStateMsg',
    '/dls2/quadruped_pympc_torques': 'dls2_msgs/msg/ControlSignalMsg',
    '/dls2/trajectory_generator': 'dls2_msgs/msg/TrajectoryGeneratorMsg',
    '/dls2/pympc_internal_state': 'dls2_msgs/msg/BlindStateMsg'
}

def get_message_type(type_string):
    try: return get_message(type_string)
    except (AttributeError, ModuleNotFoundError, ValueError): return None

def read_bag_data(bag_dir: Path):
    print(f"Opening bag: {bag_dir}")
    if not bag_dir.exists(): return None
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    reader = rosbag2_py.SequentialReader()
    try: reader.open(storage_options, converter_options)
    except Exception as e: print(f"Error opening bag: {e}", file=sys.stderr); return None

    available_topics = {}
    for meta in reader.get_all_topics_and_types():
        if meta.name in TOPIC_MAP and get_message_type(TOPIC_MAP[meta.name]):
            available_topics[meta.name] = get_message_type(TOPIC_MAP[meta.name])
    if not available_topics: print("Error: No relevant topics found.", file=sys.stderr); return None

    reader.set_filter(rosbag2_py.StorageFilter(topics=list(available_topics.keys())))
    data = defaultdict(lambda: defaultdict(list))
    t_start = None

    print("Reading messages...")
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        if t_start is None: t_start = t_ns
        msg = deserialize_message(raw, available_topics[topic])
        data[topic]['time'].append((t_ns - t_start) / 1e9)

        if topic == '/dls2/quadruped_pympc_torques':
            data[topic]['torques'].append(np.array(msg.torques))
        elif topic == '/dls2/trajectory_generator':
            data[topic]['pd_pos'].append(np.array(msg.joints_position))
            data[topic]['pd_vel'].append(np.array(msg.joints_velocity))
        elif topic == '/dls2/base_state':
            data[topic]['pos'].append(np.array(msg.position))
            data[topic]['ori'].append(np.array(msg.orientation))
            data[topic]['lin_vel'].append(np.array(msg.linear_velocity))
            data[topic]['ang_vel'].append(np.array(msg.angular_velocity))
        elif topic in ['/dls2/blind_state', '/dls2/pympc_internal_state']:
            data[topic]['qpos'].append(np.array(msg.joints_position))
            data[topic]['qvel'].append(np.array(msg.joints_velocity))
            data[topic]['effort'].append(np.array(msg.joints_effort))
            data[topic]['contact'].append(np.array(msg.feet_contact))
            data[topic]['mass_matrix'].append(np.array(msg.mass_matrix))
            data[topic]['feet_jac'].append(np.array(msg.feet_jacobians))
            data[topic]['feet_pos'].append(np.array(msg.feet_positions))
            data[topic]['hip_pos'].append(np.array(msg.hip_positions))
            data[topic]['bias_forces'].append(np.array(msg.bias_forces))
            inertia_data = [] # Default to empty
            if hasattr(msg, 'base_inertia') and len(msg.base_inertia) > 0:
                inertia_data = msg.base_inertia
            elif hasattr(msg, 'inertia') and len(msg.base_inertia) > 0:
                inertia_data = msg.base_inertia
            
            data[topic]['base_inertia'].append(np.array(inertia_data))

    print("...Done.")
    # Force aliasing if internal state is missing, so all plots use blind_state
    if '/dls2/pympc_internal_state' not in data and '/dls2/blind_state' in data:
        data['/dls2/pympc_internal_state'] = data['/dls2/blind_state']
    return data

def plot_joint_data(data):
    """FIGURE 1: Sim vs Target"""
    sim_key = '/dls2/blind_state'
    targ_key = '/dls2/trajectory_generator'
    eff_key = '/dls2/quadruped_pympc_torques' if '/dls2/quadruped_pympc_torques' in data else sim_key

    if not (sim_key in data and targ_key in data):
        print("Warning: Missing data for Figure 1.", file=sys.stderr)
        return

    print("Generating Figure 1 (Joint Data)...")
    fig, axs = plt.subplots(4, 3, figsize=(22, 16), sharex=True, num=1)
    fig.suptitle('Figure 1: Quadruped Joint State vs. Targets', fontsize=16)

    t_sim = data[sim_key]['time']
    t_targ = data[targ_key]['time']
    t_eff = data[eff_key]['time']
    eff_data = np.array(data[eff_key]['torques'] if eff_key == '/dls2/quadruped_pympc_torques' else data[eff_key]['effort'])

    legs = ['FL', 'FR', 'RL', 'RR']
    joints = ['Hip', 'Thigh', 'Calf']
    colors = ['r', 'g', 'b']

    for i, leg in enumerate(legs):
        # Pos
        ax = axs[i, 0]; ax.set_title(f'{leg} Pos')
        for j, joint in enumerate(joints):
            idx = i*3 + j
            ax.plot(t_targ, np.array(data[targ_key]['pd_pos'])[:, idx], c=colors[j], ls='--', label=f'Targ {joint}')
            ax.plot(t_sim, np.array(data[sim_key]['qpos'])[:, idx], c=colors[j], ls='-', alpha=0.7, label=f'Sim {joint}')
        ax.grid(True)
        if i==0: ax.legend(fontsize='x-small', loc='upper right')

        # Vel
        ax = axs[i, 1]; ax.set_title(f'{leg} Vel')
        for j, joint in enumerate(joints):
            idx = i*3 + j
            ax.plot(t_targ, np.array(data[targ_key]['pd_vel'])[:, idx], c=colors[j], ls='--', label=f'Targ {joint}')
            ax.plot(t_sim, np.array(data[sim_key]['qvel'])[:, idx], c=colors[j], ls='-', alpha=0.7)
        ax.grid(True)

        # Effort
        ax = axs[i, 2]; ax.set_title(f'{leg} Effort')
        if eff_data.size > 0 and eff_data.shape[1] >= 12:
             for j, joint in enumerate(joints):
                ax.plot(t_eff, eff_data[:, i*3 + j], c=colors[j], ls='-')
        else:
             ax.text(0.5, 0.5, 'No Data', ha='center', transform=ax.transAxes, color='r')
        ax.grid(True)

    fig.text(0.005, 0.5, 'Angle (rad)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.34, 0.5, 'Ang. Vel (rad/s)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.67, 0.5, 'Torque (Nm)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.5, 0.005, 'Time (s)', ha='center', fontsize=12)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

def plot_base_state(data):
    if '/dls2/base_state' not in data: return
    print("Generating Figure 2 (Base State)...")
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True, num=2)
    fig.suptitle('Figure 2: Base State', fontsize=16)
    t = data['/dls2/base_state']['time']
    axs[0].plot(t, np.array(data['/dls2/base_state']['pos'])); axs[0].set_ylabel('Pos (m)'); axs[0].legend(['X','Y','Z']); axs[0].grid(True)
    axs[1].plot(t, np.array(data['/dls2/base_state']['lin_vel'])); axs[1].set_ylabel('Lin Vel (m/s)'); axs[1].legend(['vX','vY','vZ']); axs[1].grid(True)
    axs[2].plot(t, np.array(data['/dls2/base_state']['ang_vel'])); axs[2].set_ylabel('Ang Vel (rad/s)'); axs[2].legend(['wX','wY','wZ']); axs[2].grid(True)
    axs[3].plot(t, np.array(data['/dls2/base_state']['ori'])); axs[3].set_ylabel('Quat'); axs[3].legend(['qX','qY','qZ','qW']); axs[3].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

def plot_sim_dynamics(data):
    key = '/dls2/blind_state'
    if key not in data: return
    print("Generating Figure 3 (Sim Dynamics)...")
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, num=3)
    fig.suptitle('Figure 3: Simulator Internal State', fontsize=16)
    t = data[key]['time']
    contacts = np.array(data[key]['contact'])
    #bias = np.array(data[key]['bias_forces'])
    if contacts.size > 0:
        for i, leg in enumerate(['FL','FR','RL','RR']): axs[0].plot(t, contacts[:,i]+i*1.1, label=leg)
        axs[0].set_yticks([0,1.1,2.2,3.3]); axs[0].set_yticklabels(['FL','FR','RL','RR']); axs[0].set_ylabel('Contact'); axs[0].grid(True)
    #if bias.size > 0 and bias.shape[1] >= 12:
    #    axs[1].plot(t, bias[:, 0], label='FL Hip Bias'); axs[1].plot(t, bias[:, 3], label='FR Hip Bias'); axs[1].set_ylabel('Bias (Nm)'); axs[1].legend(); axs[1].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

def plot_ctrl_dynamics(data):
    # Use aliased key, so this works even if you removed the dedicated internal topic
    key = '/dls2/pympc_internal_state'
    if key not in data or not data[key]['base_inertia'] or not data[key]['inertia'] or len(data[key]['base_inertia'][0]) == 0:
        print("Skipping Figure 4 (No rich dynamics data found).", file=sys.stderr)
        return
    print(f"Generating Figure 4 (Dynamics) from {key}...")
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, num=4)
    fig.suptitle('Figure 4: Full Dynamics Data', fontsize=16)
    t = data[key]['time']
    inertia = np.array(data[key]['base_inertia'])
    jac = np.array(data[key]['feet_jac'])
    if inertia.shape[1] == 9:
        labels = ['Ixx', 'Ixy', 'Ixz', 'Iyx', 'Iyy', 'Iyz', 'Izx', 'Izy', 'Izz']
        for i in range(9): axs[0].plot(t, inertia[:, i], label=labels[i], alpha=0.7)
        axs[0].set_ylabel('Inertia'); axs[0].legend(ncol=3, fontsize='small'); axs[0].grid(True)
    if jac.size > 0:
        axs[1].plot(t, jac, color='blue', alpha=0.05); axs[1].set_ylabel('Jacobian (All)'); axs[1].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

def plot_kinematics(data):
    # Use aliased key
    key = '/dls2/pympc_internal_state'
    # ROBUST CHECK: ensure data exists and is not empty before attempting reshape
    if key not in data or not data[key]['feet_pos'] or len(data[key]['feet_pos'][0]) == 0:
        print("Skipping Figure 5 (No Kinematics data).", file=sys.stderr)
        return
    print(f"Generating Figure 5 (Kinematics) from {key}...")
    
    t = data[key]['time']
    try:
        feet = np.array(data[key]['feet_pos']).reshape(-1, 4, 3)
        hip = np.array(data[key]['hip_pos']).reshape(-1, 4, 3)
    except ValueError as e:
        print(f"Error reshaping kinematics data: {e}. Skipping Fig 5.", file=sys.stderr)
        return

    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True, num=5)
    fig.suptitle('Figure 5: Kinematics (World Frame)', fontsize=16)
    legs = ['FL', 'FR', 'RL', 'RR']; colors = ['r', 'g', 'b', 'k']
    
    for i, leg in enumerate(legs): axs[0].plot(t, feet[:, i, 2], c=colors[i], label=leg)
    axs[0].set_title('Feet Z (Height)'); axs[0].set_ylabel('Z (m)'); axs[0].legend(loc='upper left'); axs[0].grid(True)
    for i, leg in enumerate(legs): axs[1].plot(t, hip[:, i, 2], c=colors[i], ls='--', label=f'{leg} Hip')
    axs[1].set_title('Hip Z (Height)'); axs[1].set_ylabel('Z (m)'); axs[1].legend(loc='upper left'); axs[1].grid(True)
    for i, leg in enumerate(legs): axs[2].plot(t, feet[:, i, 0], c=colors[i], label=leg)
    axs[2].set_title('Feet X'); axs[2].set_ylabel('X (m)'); axs[2].grid(True)
    for i, leg in enumerate(legs): axs[3].plot(t, feet[:, i, 1], c=colors[i], label=leg)
    axs[3].set_title('Feet Y'); axs[3].set_ylabel('Y (m)'); axs[3].set_xlabel('Time (s)'); axs[3].grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_dir", type=Path)
    args = parser.parse_args()
    data = read_bag_data(args.bag_dir)
    if data:
        plot_joint_data(data)
        plot_base_state(data)
        plot_sim_dynamics(data)
        plot_ctrl_dynamics(data)
        plot_kinematics(data)
        print("Showing plots. Close windows to exit.")
        plt.show()

if __name__ == "__main__":
    main()