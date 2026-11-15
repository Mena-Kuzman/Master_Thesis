"""Isaac-side HeightMap adapter

Provides a HeightMap-compatible object (same API as gym_quadruped.sensors.heightmap.HeightMap)
but uses an external raycast function (callback) to query terrain in Isaac Sim.

Usage:
  - Provide a `raycast_fn(origin: (x,y,z), direction: (dx,dy,dz), max_dist) -> (x,y,z)` callable
    that performs a physics raycast in Isaac and returns the intersection point (or [-1,-1,-1]).
  - Instantiate HeightMapIsaac with the same grid parameters and pass it into your controller
    in place of the MuJoCo HeightMap.
"""
from __future__ import annotations

import time
import os
import numpy as np
from typing import Callable, Optional, List


class HeightMapIsaac:
    """HeightMap adapter that matches gym_quadruped.sensors.heightmap.HeightMap API.

    Key methods/properties implemented:
      - update_height_map(center, yaw)
      - get_height(target)
      - get_sparse_samples(offsets) -> list of heights  <-- NEW
      - .data (numpy array shape: rows x cols x 1 x 3)
      - .geom_ids (rows x cols int array, for visualization handles)
      - .last_sim_time, .num_rows, .num_cols
      - .n : convenience integer used in some parts of the code (set to num_rows)
    """

    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        dist_x: float,
        dist_y: float,
        raycast_fn: Optional[Callable[[list, list, float], list]] = None,
        raycast_max_dist: float = 1.0,
        height_bias: float = 0.0,
    ) -> None:
        self.use_map_initialization = False
        self.last_time = 0.0
        self.data = None

        self.num_rows = int(num_rows)
        self.num_cols = int(num_cols)
        self.n = int(self.num_rows)

        self.geom_ids = -np.ones((self.num_rows, self.num_cols), dtype=np.int32)

        self.dist_x = float(dist_x)
        self.dist_y = float(dist_y)
        self._local_offsets = self._build_local_offsets()
        self.sensor_matrix = np.empty((self.num_rows, self.num_cols, 1, 3))
        self.sensor_data_matrix = np.full((self.num_rows, self.num_cols, 1, 3), np.nan, dtype=np.float64)
        self.old_sensor_data_matrix = np.full((self.num_rows, self.num_cols, 1, 3), np.nan, dtype=np.float64)

        self._raycast_fn = raycast_fn
        self._raycast_max_dist = float(raycast_max_dist)
        self._height_bias = float(height_bias)
        self.last_raycast_stats = None
        self.is_valid = False
        self._min_valid_ratio = 0.35
        self._min_valid_points = 8
        self._debug_enabled = os.environ.get("HEIGHTMAP_ISAAC_DEBUG", "0") == "1"
        self._last_debug_print_time = 0.0
        self._raw_debug_prints = 0
        self._last_raw_debug_time = 0.0

    def _build_local_offsets(self) -> np.ndarray:
        half_x = (self.num_rows - 1) / 2.0
        half_y = (self.num_cols - 1) / 2.0
        xs = (np.arange(self.num_rows, dtype=np.float64) - half_x) * self.dist_x
        ys = (np.arange(self.num_cols, dtype=np.float64) - half_y) * self.dist_y
        xg, yg = np.meshgrid(xs, ys, indexing='ij')
        return np.stack((xg, yg), axis=-1)

    @property
    def last_sim_time(self) -> float:
        return float(self.last_time)

    @last_sim_time.setter
    def last_sim_time(self, t: float) -> None:
        self.last_time = float(t)

    def raycast_sensor(self, pos, dist=1.0):
        if self._raycast_fn is None:
            raise RuntimeError("No raycast function provided to HeightMapIsaac")

        origin = [float(pos[0]), float(pos[1]), float(pos[2])]
        direction = [0.0, 0.0, -1.0]
        hit = self._raycast_fn(origin, direction, float(dist))
        if hit is None:
            return [-1.0, -1.0, -1.0]
        return hit

    def _is_valid_hit(self, pos_data):
        if not isinstance(pos_data, (list, tuple)) or len(pos_data) < 3:
            return False, None
        try:
            xh, yh, zh = float(pos_data[0]), float(pos_data[1]), float(pos_data[2])
        except Exception:
            return False, None
        if (not np.isfinite(xh)) or (not np.isfinite(yh)) or (not np.isfinite(zh)):
            return False, None
        if zh <= -0.999:
            return False, None
        return True, [xh, yh, zh]

    def create_sensor_matrix(self, center, yaw=0.0):
        R = np.array([np.cos(yaw), np.sin(yaw), -np.sin(yaw), np.cos(yaw)], dtype=np.float64).reshape((2, 2))

        ref_x, ref_y = float(center[0]), float(center[1])
        ref_z = float(center[2]) + 0.6

        rotated = self._local_offsets @ R
        self.sensor_matrix[:, :, 0, 0] = ref_x + rotated[:, :, 0]
        self.sensor_matrix[:, :, 0, 1] = ref_y + rotated[:, :, 1]
        self.sensor_matrix[:, :, 0, 2] = ref_z

        self.sensor_data_matrix = self.old_sensor_data_matrix.copy()
        raycast_fn = self._raycast_fn
        max_dist = float(self._raycast_max_dist)

        hits = 0
        z_vals = []

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                origin = list(self.sensor_matrix[i, j, 0])
                try:
                    if raycast_fn is None:
                        raise RuntimeError("No raycast function provided to HeightMapIsaac")
                    pos_data = raycast_fn(origin, [0.0, 0.0, -1.0], max_dist)
                except Exception:
                    continue

                valid, hit = self._is_valid_hit(pos_data)
                if not valid:
                    continue

                self.sensor_data_matrix[i, j, 0, 0] = hit[0]
                self.sensor_data_matrix[i, j, 0, 1] = hit[1]
                self.sensor_data_matrix[i, j, 0, 2] = hit[2]
                hits += 1
                z_vals.append(hit[2])

        finite_z_mask = np.isfinite(self.sensor_data_matrix[:, :, 0, 2])
        valid_points = int(np.count_nonzero(finite_z_mask))
        total_points = int(self.num_rows * self.num_cols)
        valid_ratio = (valid_points / total_points) if total_points > 0 else 0.0

        self.last_raycast_stats = {
            'hits': hits,
            'total': total_points,
            'valid_points': valid_points,
            'valid_ratio': float(valid_ratio),
        }
        if z_vals:
            self.last_raycast_stats['min_z'] = float(min(z_vals))
            self.last_raycast_stats['max_z'] = float(max(z_vals))
            self.last_raycast_stats['mean_z'] = float(np.mean(z_vals))

        self.is_valid = (valid_points >= self._min_valid_points) and (valid_ratio >= self._min_valid_ratio)

        if self._debug_enabled:
            now = time.time()
            if (now - self._last_debug_print_time) > 0.5:
                self._last_debug_print_time = now
                try:
                    print(f"[HeightMapIsaac] hits={hits}/{total_points} valid={valid_points} ratio={valid_ratio:.2f} is_valid={self.is_valid}")
                except Exception:
                    pass

        self.old_sensor_data_matrix = self.sensor_data_matrix.copy()
        return self.sensor_data_matrix

    def update_height_map(self, center, yaw=0.0):
        latest = self.create_sensor_matrix(center, yaw)
        if self.is_valid:
            self.data = latest
        elif self.data is None:
            self.data = None
        return self.data

    def get_height(self, target):
        try:
            if self.data is None:
                return None

            pts = self.data.reshape((-1, 3))[:, 0:3]
            finite_mask = np.isfinite(pts).all(axis=1)
            pts = pts[finite_mask]
            if pts.size == 0:
                return None

            deltas = pts[:, 0:2] - np.array(target[0:2], dtype=np.float64)
            d2 = np.sum(deltas * deltas, axis=1)
            idx = int(np.argmin(d2))
            z = float(pts[idx, 2])
            return float(z + self._height_bias)
        except Exception:
            return None

    def get_sparse_samples(self, target_xy: list, offsets: List[tuple], fallback_z: float) -> List[float]:
        """
        Samples the heightmap at specific (dx, dy) offsets relative to the target_xy.
        If a sample falls outside the valid grid, it returns fallback_z.
        
        Args:
            target_xy: The [x, y] world coordinate of the foot center.
            offsets: List of tuples [(dx1, dy1), (dx2, dy2), ...] in meters.
            fallback_z: Value to return if a point is invalid or missing.
            
        Returns:
            List of z-heights corresponding to the offsets.
        """
        results = []
        for dx, dy in offsets:
            sample_pt = [target_xy[0] + dx, target_xy[1] + dy, 0.0]
            z = self.get_height(sample_pt)
            if z is None:
                results.append(fallback_z)
            else:
                results.append(z)
        return results