#!/usr/bin/env python3
import csv
import os
import sys
import argparse
from collections import defaultdict
import numpy as np
import time
import re

# Ensure IsaacLabFromTemu dir is on sys.path so we can import compare_mpc_vs_mpc_rl
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ISAACLAB_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if ISAACLAB_DIR not in sys.path:
    sys.path.insert(0, ISAACLAB_DIR)

# Import plotting helper from the compare script
from compare_mpc_vs_mpc_rl import _plot_success_summary_by_speed_and_height


def read_csv_rows(csv_path):
    candidate_paths = []
    csv_path = os.path.expanduser(csv_path)
    if os.path.isabs(csv_path):
        candidate_paths.append(csv_path)
    else:
        candidate_paths.append(os.path.abspath(csv_path))
        candidate_paths.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..', csv_path)))
        candidate_paths.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', csv_path)))

    resolved_csv_path = None
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            resolved_csv_path = candidate
            break

    if resolved_csv_path is None:
        raise FileNotFoundError(f"Could not find CSV file. Tried: {candidate_paths}")

    rows = []
    with open(resolved_csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # normalize types
            row_parsed = {
                'terrain': row.get('terrain', ''),
                'command_forward': float(row.get('command_forward', 0.0)),
                'stair_step_height': float(row.get('stair_step_height', 0.0)),
                'case': row.get('case', ''),
                'episode_idx': int(float(row.get('episode_idx', -1))) if row.get('episode_idx','')!='' else -1,
                'course_success': float(row.get('course_success', '0') or 0.0),
            }
            rows.append(row_parsed)
    return rows


def build_series(rows, terrain):
    # unique sorted speeds and heights
    speeds = sorted({float(r['command_forward']) for r in rows if r['terrain'] == terrain})
    heights = sorted({float(r['stair_step_height']) for r in rows if r['terrain'] == terrain})

    modes = sorted({r['case'] for r in rows if r['terrain'] == terrain})

    # Prepare containers
    speed_success_rate_by_mode = {m: {} for m in modes}
    speed_success_counts_by_mode = {m: {} for m in modes}
    height_success_rate_by_mode = {m: {} for m in modes}
    height_success_counts_by_mode = {m: {} for m in modes}

    def mean_metric(terrain, speed, case, metric, stair_step_height=None):
        vals = [r[metric] for r in rows if r['terrain'] == terrain and r['case'] == case and abs(float(r['command_forward']) - float(speed)) < 1e-9 and (stair_step_height is None or abs(float(r['stair_step_height']) - float(stair_step_height)) < 1e-9)]
        if not vals:
            return float('nan')
        return float(np.mean(vals))

    for m in modes:
        for h in heights:
            label = f"{h:.2f} m"
            vals = [mean_metric(terrain, s, m, 'course_success', stair_step_height=h) for s in speeds]
            speed_success_rate_by_mode[m][label] = vals
            # counts per speed for this height
            counts = []
            for s in speeds:
                matching = [r for r in rows if r['terrain'] == terrain and r['case'] == m and abs(float(r['command_forward']) - float(s)) < 1e-9 and abs(float(r['stair_step_height']) - float(h)) < 1e-9]
                counts.append(int(sum(1 for r in matching if float(r['course_success']) > 0.5)))
            speed_success_counts_by_mode[m][label] = counts

        for s in speeds:
            label = f"{s:.2f} m/s"
            vals = [mean_metric(terrain, s, m, 'course_success', stair_step_height=h) for h in heights]
            height_success_rate_by_mode[m][label] = vals
            counts = []
            for h in heights:
                matching = [r for r in rows if r['terrain'] == terrain and r['case'] == m and abs(float(r['command_forward']) - float(s)) < 1e-9 and abs(float(r['stair_step_height']) - float(h)) < 1e-9]
                counts.append(int(sum(1 for r in matching if float(r['course_success']) > 0.5)))
            height_success_counts_by_mode[m][label] = counts

    return speeds, heights, speed_success_rate_by_mode, speed_success_counts_by_mode, height_success_rate_by_mode, height_success_counts_by_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='Path to go2_mpc_vs_mpc_rl_summary_*.csv')
    parser.add_argument('--terrain', choices=['stairs','procedural'], default='stairs')
    parser.add_argument('--output-dir', default='analysis')
    args = parser.parse_args()

    rows = read_csv_rows(args.csv)
    if not rows:
        print('No rows read from CSV')
        return

    os.makedirs(args.output_dir, exist_ok=True)

    (
        speeds,
        heights,
        speed_rate_by_mode,
        speed_counts_by_mode,
        height_rate_by_mode,
        height_counts_by_mode,
    ) = build_series(rows, args.terrain)

    csv_name = os.path.basename(os.path.expanduser(args.csv))

    match = re.search(r"go2_mpc_vs_mpc_rl_summary_(\d+)\.csv$", csv_name)
    if match:
        stamp = match.group(1)
        out_success_summary = os.path.join(
            args.output_dir,
            f"go2_mpc_vs_mpc_rl_{args.terrain}_success_summary_speed_height_{stamp}.png",
        )
        out_success_counts = os.path.join(
            args.output_dir,
            f"go2_mpc_vs_mpc_rl_{args.terrain}_successful_episodes_speed_height_{stamp}.png",
        )
    else:
        stamp = str(int(time.time()))
        out_success_summary = os.path.join(args.output_dir, f"fromcsv_{csv_name}_{args.terrain}_success_summary_{stamp}.png")
        out_success_counts = os.path.join(args.output_dir, f"fromcsv_{csv_name}_{args.terrain}_successful_episodes_{stamp}.png")

    _plot_success_summary_by_speed_and_height(
        out_success_summary,
        args.terrain,
        speeds,
        heights,
        speed_rate_by_mode,
        height_rate_by_mode,
        ylabel='Success rate',
        title='Success rate by speed and stair height',
        value_format='{:.2f}',
    )

    _plot_success_summary_by_speed_and_height(
        out_success_counts,
        args.terrain,
        speeds,
        heights,
        speed_counts_by_mode,
        height_counts_by_mode,
        ylabel='Successful episodes',
        title='Successful episodes by speed, mode, and stair height',
        value_format='{:.0f}',
    )

    print('Saved plots to', args.output_dir)


if __name__ == '__main__':
    main()
