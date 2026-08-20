#!/usr/bin/env python3
import sys
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_vector(s):
    """
    Cleans and parses a string of numbers.
    Removes brackets and splits by whitespace.
    """
    s = s.strip().replace('[', '').replace(']', '')
    if not s:
        return []
    try:
        # --- THIS IS THE FIX ---
        # Use s.split() which correctly handles multiple whitespace characters
        # and add 'if v' to filter out any remaining empty strings.
        return [float(v) for v in s.split() if v]
        # ---------------------
    except ValueError as e:
        # This will now print the *cleaned* string, which is more useful
        print(f"Error parsing vector: '{s}' -> {e}", file=sys.stderr)
        return []

def main():
    parser = argparse.ArgumentParser(description="Parse and plot Quadruped_PyMPC_Node logs from a file.")
    parser.add_argument("logfile", type=str, help="Path to the log file to parse.")
    args = parser.parse_args()

    data = {
        'time': [],
        'actual_pos': defaultdict(list),
        'target_pos': defaultdict(list),
        'actual_vel': defaultdict(list),
        'base_vel': defaultdict(list)
    }
    
    # --- REGEX TO DETECT START OF A BLOCK ---
    re_qpos_start = re.compile(r"Joint Pos \(qpos\[7:\]\):")
    re_qvel_start = re.compile(r"Joint Vel \(qvel\[6:\]\):")
    re_base_vel_start = re.compile(r"Base Vel \(qvel\[0:6\]\):")
    
    # --- REGEX TO CAPTURE CONTENT ---
    # These will capture *only* the number string, with or without brackets
    re_qpos = re.compile(r"Joint Pos \(qpos\[7:\]\): \[?\s*(.*)")
    re_qvel = re.compile(r"Joint Vel \(qvel\[6:\]\): \[?\s*(.*)")
    re_base_vel = re.compile(r"Base Vel \(qvel\[0:6\]\): \[?\s*(.*)")
    
    # These PD target regexes are multi-line and should be fine
    re_pd_pos_header = re.compile(r"PD Target Pos \(FL/FR/RL/RR\):")
    re_pd_pos_fl_fr = re.compile(r"\s+FL: \[(.*?)\] \| FR: \[(.*?)\]")
    re_pd_pos_rl_rr = re.compile(r"\s+RL: \[(.*?)\] \| RR: \[(.*?)\]")
    # ----------------------------------------------------------------

    # State machine variables
    parsing_pd_pos = False
    fl_pos_buf, fr_pos_buf = [], []
    
    parsing_qpos = False
    parsing_qvel = False
    parsing_base_vel = False
    
    qpos_buffer = ""
    qvel_buffer = ""
    base_vel_buffer = ""
    
    step_counter = 0

    print(f"Parsing log file: {args.logfile}", file=sys.stderr)

    try:
        with open(args.logfile, 'r') as f:
            for line in f:
                # Get the part of the log after the [Node]: timestamp
                parts = line.split("Node]:", 1)
                if len(parts) < 2:
                    # This line is probably a continuation of a multi-line vector
                    clean_line = line
                else:
                    # This is a new log line
                    clean_line = parts[1]

                # --- State Machine for PD Targets (multi-line) ---
                if re_pd_pos_header.search(clean_line):
                    parsing_pd_pos = True
                    continue

                if parsing_pd_pos:
                    match = re_pd_pos_fl_fr.search(clean_line)
                    if match:
                        fl_pos_buf = parse_vector(match.group(1))
                        fr_pos_buf = parse_vector(match.group(2))
                        continue
                    
                    match = re_pd_pos_rl_rr.search(clean_line)
                    if match:
                        rl_pos = parse_vector(match.group(1))
                        rr_pos = parse_vector(match.group(2))
                        
                        if fl_pos_buf and fr_pos_buf and rl_pos and rr_pos:
                            data['target_pos']['FL_hip'].append(fl_pos_buf[0])
                            data['target_pos']['FL_thigh'].append(fl_pos_buf[1])
                            data['target_pos']['FL_calf'].append(fl_pos_buf[2])
                            data['target_pos']['FR_hip'].append(fr_pos_buf[0])
                            data['target_pos']['FR_thigh'].append(fr_pos_buf[1])
                            data['target_pos']['FR_calf'].append(fr_pos_buf[2])
                            data['target_pos']['RL_hip'].append(rl_pos[0])
                            data['target_pos']['RL_thigh'].append(rl_pos[1])
                            data['target_pos']['RL_calf'].append(rl_pos[2])
                            data['target_pos']['RR_hip'].append(rr_pos[0])
                            data['target_pos']['RR_thigh'].append(rr_pos[1])
                            data['target_pos']['RR_calf'].append(rr_pos[2])
                        
                        fl_pos_buf, fr_pos_buf = [], []
                        parsing_pd_pos = False
                        continue
                
                # --- State Machine for qpos ---
                match = re_qpos.search(clean_line)
                if match:
                    parsing_qpos = True
                    qpos_buffer = match.group(1) # Get everything after the label
                elif parsing_qpos:
                    qpos_buffer += clean_line # Append the wrapped line

                if parsing_qpos and ']' in qpos_buffer:
                    parsing_qpos = False
                    vals = parse_vector(qpos_buffer)
                    if len(vals) == 12:
                        data['actual_pos']['FL_hip'].append(vals[0])
                        data['actual_pos']['FL_thigh'].append(vals[1])
                        data['actual_pos']['FL_calf'].append(vals[2])
                        data['actual_pos']['FR_hip'].append(vals[3])
                        data['actual_pos']['FR_thigh'].append(vals[4])
                        data['actual_pos']['FR_calf'].append(vals[5])
                        data['actual_pos']['RL_hip'].append(vals[6])
                        data['actual_pos']['RL_thigh'].append(vals[7])
                        data['actual_pos']['RL_calf'].append(vals[8])
                        data['actual_pos']['RR_hip'].append(vals[9])
                        data['actual_pos']['RR_thigh'].append(vals[10])
                        data['actual_pos']['RR_calf'].append(vals[11])
                        
                        data['time'].append(step_counter)
                        step_counter += 1
                    qpos_buffer = "" # Clear buffer
                    continue

                # --- State Machine for base_vel ---
                match = re_base_vel.search(clean_line)
                if match:
                    parsing_base_vel = True
                    base_vel_buffer = match.group(1)
                elif parsing_base_vel:
                    base_vel_buffer += clean_line
                
                if parsing_base_vel and ']' in base_vel_buffer:
                    parsing_base_vel = False
                    vals = parse_vector(base_vel_buffer)
                    if len(vals) == 6:
                        data['base_vel']['vx'].append(vals[0])
                        data['base_vel']['vy'].append(vals[1])
                        data['base_vel']['vz'].append(vals[2])
                        data['base_vel']['wx'].append(vals[3])
                        data['base_vel']['wy'].append(vals[4])
                        data['base_vel']['wz'].append(vals[5])
                    base_vel_buffer = ""
                    continue
                
                # --- State Machine for qvel ---
                match = re_qvel.search(clean_line)
                if match:
                    parsing_qvel = True
                    qvel_buffer = match.group(1)
                elif parsing_qvel:
                    qvel_buffer += clean_line

                if parsing_qvel and ']' in qvel_buffer:
                    parsing_qvel = False
                    vals = parse_vector(qvel_buffer)
                    if len(vals) == 12:
                        data['actual_vel']['FL_hip'].append(vals[0])
                        data['actual_vel']['FL_thigh'].append(vals[1])
                        data['actual_vel']['FL_calf'].append(vals[2])
                        # ... add others if needed
                    qvel_buffer = ""
                    continue
        
    except KeyboardInterrupt:
        print("\nPlotting stopped.", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Log file not found at {args.logfile}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
    
    # --- Plotting ---
    if not data['time']:
        print("No data was successfully parsed. Check log file format and regex.", file=sys.stderr)
        return

    # Ensure all lists are the same length as time, pad if necessary
    N = len(data['time'])
    for category in ['actual_pos', 'target_pos', 'base_vel', 'actual_vel']:
        for key in data[category].keys():
            if len(data[category][key]) == 0:
                data[category][key] = [0] * N
            elif len(data[category][key]) < N:
                padding = [data[category][key][-1]] * (N - len(data[category][key]))
                data[category][key].extend(padding)
            data[category][key] = data[category][key][:N]

    time_steps = data['time']
    
    fig, axs = plt.subplots(6, 1, figsize=(15, 24), sharex=True)
    
    # Plot 1: FL Leg Position
    axs[0].plot(time_steps, data['actual_pos']['FL_hip'], 'r-', label='Actual Hip (qpos)')
    axs[0].plot(time_steps, data['target_pos']['FL_hip'], 'r--', label='Target Hip (PD)')
    axs[0].plot(time_steps, data['actual_pos']['FL_thigh'], 'g-', label='Actual Thigh (qpos)')
    axs[0].plot(time_steps, data['target_pos']['FL_thigh'], 'g--', label='Target Thigh (PD)')
    axs[0].plot(time_steps, data['actual_pos']['FL_calf'], 'b-', label='Actual Calf (qpos)')
    axs[0].plot(time_steps, data['target_pos']['FL_calf'], 'b--', label='Target Calf (PD)')
    axs[0].set_title('Front-Left (FL) Leg Position')
    axs[0].set_ylabel('Angle (rad)')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    
    # Plot 2: FR Leg Position
    axs[1].plot(time_steps, data['actual_pos']['FR_hip'], 'r-', label='Actual Hip (qpos)')
    axs[1].plot(time_steps, data['target_pos']['FR_hip'], 'r--', label='Target Hip (PD)')
    axs[1].plot(time_steps, data['actual_pos']['FR_thigh'], 'g-', label='Actual Thigh (qpos)')
    axs[1].plot(time_steps, data['target_pos']['FR_thigh'], 'g--', label='Target Thigh (PD)')
    axs[1].plot(time_steps, data['actual_pos']['FR_calf'], 'b-', label='Actual Calf (qpos)')
    axs[1].plot(time_steps, data['target_pos']['FR_calf'], 'b--', label='Target Calf (PD)')
    axs[1].set_title('Front-Right (FR) Leg Position')
    axs[1].set_ylabel('Angle (rad)')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)
    
    # Plot 3: RL Leg Position
    axs[2].plot(time_steps, data['actual_pos']['RL_hip'], 'r-', label='Actual Hip (qpos)')
    axs[2].plot(time_steps, data['target_pos']['RL_hip'], 'r--', label='Target Hip (PD)')
    axs[2].plot(time_steps, data['actual_pos']['RL_thigh'], 'g-', label='Actual Thigh (qpos)')
    axs[2].plot(time_steps, data['target_pos']['RL_thigh'], 'g--', label='Target Thigh (PD)')
    axs[2].plot(time_steps, data['actual_pos']['RL_calf'], 'b-', label='Actual Calf (qpos)')
    axs[2].plot(time_steps, data['target_pos']['RL_calf'], 'b--', label='Target Calf (PD)')
    axs[2].set_title('Rear-Left (RL) Leg Position')
    axs[2].set_ylabel('Angle (rad)')
    axs[2].legend(loc='upper right')
    axs[2].grid(True)
    
    # Plot 4: RR Leg Position
    axs[3].plot(time_steps, data['actual_pos']['RR_hip'], 'r-', label='Actual Hip (qpos)')
    axs[3].plot(time_steps, data['target_pos']['RR_hip'], 'r--', label='Target Hip (PD)')
    axs[3].plot(time_steps, data['actual_pos']['RR_thigh'], 'g-', label='Actual Thigh (qpos)')
    axs[3].plot(time_steps, data['target_pos']['RR_thigh'], 'g--', label='Target Thigh (PD)')
    axs[3].plot(time_steps, data['actual_pos']['RR_calf'], 'b-', label='Actual Calf (qpos)')
    axs[3].plot(time_steps, data['target_pos']['RR_calf'], 'b--', label='Target Calf (PD)')
    axs[3].set_title('Rear-Right (RR) Leg Position')
    axs[3].set_ylabel('Angle (rad)')
    axs[3].legend(loc='upper right')
    axs[3].grid(True)
    
    # Plot 5: Base Velocity
    axs[4].plot(time_steps, data['base_vel']['vx'], label='Base vx (m/s)')
    axs[4].plot(time_steps, data['base_vel']['vy'], label='Base vy (m/s)')
    axs[4].plot(time_steps, data['base_vel']['vz'], label='Base vz (m/s)')
    axs[4].plot(time_steps, data['base_vel']['wz'], label='Base wz (rad/s)', linestyle=':')
    axs[4].set_title('Base Velocity (qvel[0:6]) - PROOF OF INSTABILITY')
    axs[4].set_ylabel('Velocity')
    axs[4].legend(loc='upper right')
    axs[4].grid(True)

    # Plot 6: FL Joint Velocity
    axs[5].plot(time_steps, data['actual_vel']['FL_hip'], label='Actual Hip Vel (qvel)')
    axs[5].plot(time_steps, data['actual_vel']['FL_thigh'], label='Actual Thigh Vel (qvel)')
    axs[5].plot(time_steps, data['actual_vel']['FL_calf'], label='Actual Calf Vel (qvel)')
    axs[5].set_title('FL Joint Velocities (qvel[6:])')
    axs[5].set_xlabel('Time Step')
    axs[5].set_ylabel('Ang. Velocity (rad/s)')
    axs[5].legend(loc='upper right')
    axs[5].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()