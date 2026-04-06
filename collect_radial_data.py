#!/usr/bin/env python3
"""
collect_radial_data.py  –  Radial-sweep data collection for XArm fingerpad

Samples a uniform grid of starting points across the sensor rectangle (2 mm apart
by default).  From each starting point the arm moves outward in several randomly-
chosen directions, all the way to the rectangle edge, then returns to the starting
point.  The number of directions sampled per point is proportional to the available
angular range at that point:

  Interior point   →  360 °  →  ~36 directions
  Edge point       →  180 °  →  ~18 directions
  Corner point     →   90 °  →   ~9 directions

A random speed in [--vmin … --vmax] mm/s is chosen independently for every move.

Folder layout
─────────────
  {output}/
    pt{IIII}_dir{JJJ}/          e.g.  pt0042_dir003/
        robot_pose_planned.csv
        robot_pose_recorded.csv
        motion_parameters.csv   (point_x_mm, point_z_mm, direction_deg, speed_mm_s,
                                  edge_x_mm, edge_z_mm, distance_mm)
        audio.wav               (only with --audio)
        audio_sync.json
        audio_timestamps.csv

Usage
─────
  # Dry-run preview:
  python collect_radial_data.py -o ./data --dry-run

  # Live robot:
  python collect_radial_data.py -o ./data --ip 192.168.1.200 --audio

  # Custom grid spacing and speed range:
  python collect_radial_data.py -o ./data --grid-mm 3 --vmin 5 --vmax 60
"""

import argparse
import csv
import json
import math
import os
import queue as _audio_queue
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    sys.exit("NumPy required:  pip install numpy")

try:
    from xarm.wrapper import XArmAPI
    XARM_AVAILABLE = True
except ImportError:
    XARM_AVAILABLE = False

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Hardware limits
# ─────────────────────────────────────────────────────────────────────────────
XARM_MAX_LIN_MM_S  = 1000.0
XARM_MAX_ACC_MM_S2 = 5000.0

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_GRID_MM     = 2.0
DEFAULT_VMIN_MM_S   = 10.0
DEFAULT_VMAX_MM_S   = 50.0
DEFAULT_MVACC_MM_S2 = 200.0
DEFAULT_DWELL_S     = 0.3
DEFAULT_MARGIN      = 0.05
EDGE_TOL_MM         = 1e-3   # distance to wall to be considered "on edge"
DEG_PER_DIRECTION   = 10.0   # one direction sampled per this many degrees of range

SEP = "─" * 64


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_valid_arc(px: float, pz: float, hw: float, hh: float) -> tuple[float, float]:
    """
    Return the contiguous valid angular arc as (theta_lo_deg, arc_deg).

    Convention: theta=0 → +X, theta=90 → +Z (CCW).
    "Valid" means the direction leads immediately into the rectangle interior.

    Strategy: each active wall contributes an inward-facing normal.  The valid
    arc is a half-plane (180°) per wall, intersected for corners (→ 90°).
    We compute the arc centre as the vector mean of the inward normals.

      Interior  →  360°  (lo=0)
      Edge      →  180°  (lo = normal_dir − 90°)
      Corner    →   90°  (lo = mean_normal − 45°)
    """
    on_left   = abs(px + hw) < EDGE_TOL_MM
    on_right  = abs(px - hw) < EDGE_TOL_MM
    on_bottom = abs(pz + hh) < EDGE_TOL_MM
    on_top    = abs(pz - hh) < EDGE_TOL_MM

    # Inward-facing normals of each active wall (degrees)
    #   left wall  → inward = +X = 0°
    #   right wall → inward = -X = 180°
    #   bottom     → inward = +Z = 90°
    #   top        → inward = -Z = 270°
    normals = []
    if on_left:   normals.append(0.0)
    if on_right:  normals.append(180.0)
    if on_bottom: normals.append(90.0)
    if on_top:    normals.append(270.0)

    if not normals:
        return 0.0, 360.0   # interior point: all directions valid

    # arc width = 360° / 2^(number of active walls)
    arc_deg = 360.0 / (2 ** len(normals))   # 180° for 1 wall, 90° for 2 walls

    if len(normals) == 1:
        center = normals[0]
    else:
        # Vector mean of inward normals → gives the bisecting direction
        rads = [math.radians(n) for n in normals]
        cx = sum(math.cos(r) for r in rads)
        cz = sum(math.sin(r) for r in rads)
        center = math.degrees(math.atan2(cz, cx)) % 360.0

    lo = (center - arc_deg / 2.0) % 360.0
    return lo, arc_deg


def ray_to_edge(px: float, pz: float, theta_deg: float,
                hw: float, hh: float) -> tuple[float, float]:
    """
    From point (px, pz) shoot a ray in direction theta_deg (degrees, CCW from +X).
    Return the point where it first hits the rectangle boundary.
    """
    theta = math.radians(theta_deg)
    dx, dz = math.cos(theta), math.sin(theta)
    t_candidates = []
    if abs(dx) > 1e-12:
        for wall_x in (-hw, hw):
            t = (wall_x - px) / dx
            if t > 1e-9:
                z_hit = pz + t * dz
                if -hh - 1e-9 <= z_hit <= hh + 1e-9:
                    t_candidates.append(t)
    if abs(dz) > 1e-12:
        for wall_z in (-hh, hh):
            t = (wall_z - pz) / dz
            if t > 1e-9:
                x_hit = px + t * dx
                if -hw - 1e-9 <= x_hit <= hw + 1e-9:
                    t_candidates.append(t)
    if not t_candidates:
        return px, pz  # degenerate – shouldn't happen for interior points
    t_min = min(t_candidates)
    ex = np.clip(px + t_min * dx, -hw, hw)
    ez = np.clip(pz + t_min * dz, -hh, hh)
    return float(ex), float(ez)


def sample_directions(lo_deg: float, arc_deg: float, n: int, rng: random.Random) -> list[float]:
    """
    Stratified-random sample of n angles within the arc [lo_deg, lo_deg+arc_deg).
    Divides the arc into n equal bins and picks one uniform random angle per bin.
    """
    if n <= 0:
        return []
    bin_size = arc_deg / n
    return [(lo_deg + (k + rng.random()) * bin_size) % 360.0 for k in range(n)]


def build_grid(hw: float, hh: float, step: float) -> list[tuple[float, float]]:
    """Return all grid points in [-hw,hw]×[-hh,hh] spaced step mm apart."""
    xs = np.arange(-hw, hw + step * 0.5, step)
    zs = np.arange(-hh, hh + step * 0.5, step)
    points = []
    for z in zs:
        for x in xs:
            x_c = float(np.clip(x, -hw, hw))
            z_c = float(np.clip(z, -hh, hh))
            points.append((x_c, z_c))
    return points


# ─────────────────────────────────────────────────────────────────────────────
# Plan all motions up-front
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MotionPlan:
    pt_idx:      int
    dir_idx:     int
    px:          float   # starting point in sensor frame (mm)
    pz:          float
    theta_deg:   float   # direction of outward stroke
    speed_mm_s:  float
    edge_x:      float   # endpoint on rectangle boundary
    edge_z:      float
    distance_mm: float   # one-way distance
    folder:      str     # e.g.  "pt0042_dir003"


def plan_all(hw: float, hh: float, grid_mm: float,
             vmin: float, vmax: float,
             seed: int = 42) -> list[MotionPlan]:
    rng = random.Random(seed)
    grid = build_grid(hw, hh, grid_mm)
    plans = []
    for pt_idx, (px, pz) in enumerate(grid):
        lo, arc = compute_valid_arc(px, pz, hw, hh)
        n_dirs  = max(1, round(arc / DEG_PER_DIRECTION))
        dirs    = sample_directions(lo, arc, n_dirs, rng)
        for dir_idx, theta in enumerate(dirs):
            ex, ez   = ray_to_edge(px, pz, theta, hw, hh)
            dist     = math.hypot(ex - px, ez - pz)
            if dist < 0.1:          # ray hit boundary without moving – skip
                continue
            speed    = rng.uniform(vmin, vmax)
            folder   = f"pt{pt_idx:04d}_dir{dir_idx:03d}"
            plans.append(MotionPlan(
                pt_idx=pt_idx, dir_idx=dir_idx,
                px=px, pz=pz,
                theta_deg=theta,
                speed_mm_s=speed,
                edge_x=ex, edge_z=ez,
                distance_mm=dist,
                folder=folder,
            ))
    return plans


# ─────────────────────────────────────────────────────────────────────────────
# Audio recorder  (identical to collect_sweep_data.py)
# ─────────────────────────────────────────────────────────────────────────────

def find_xr18_input(name_hint="XR18"):
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if name_hint.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return i, d["max_input_channels"]
    raise RuntimeError(f"XR18 input device not found (hint='{name_hint}')")


class AudioRecorder:
    CHECKPOINT_INTERVAL_S = 1.0

    def __init__(self, folder, sr=48000, blocksize=2048, device_hint="XR18"):
        self._folder      = folder
        self._sr          = sr
        self._blocksize   = blocksize
        self._device_hint = device_hint
        self._wav_path    = os.path.join(folder, "audio.wav")
        self._sync_path   = os.path.join(folder, "audio_sync.json")
        self._ckpt_path   = os.path.join(folder, "audio_timestamps.csv")
        self._q           = _audio_queue.Queue(maxsize=256)
        self._stop_flag   = threading.Event()
        self._stream      = None
        self._writer_thr: Optional[threading.Thread] = None
        self._ckpt_thr:   Optional[threading.Thread] = None
        self._sample_count = 0
        self._start_epoch  = 0.0
        self._in_ch        = 0

    def start(self):
        os.makedirs(self._folder, exist_ok=True)
        in_dev_idx, in_ch = find_xr18_input(self._device_hint)
        self._in_ch = in_ch
        wav_path, sr = self._wav_path, self._sr
        stop_flag, q = self._stop_flag, self._q

        def _writer():
            with sf.SoundFile(wav_path, mode="w", samplerate=sr,
                              channels=in_ch, subtype="FLOAT") as f:
                while not (stop_flag.is_set() and q.empty()):
                    try:
                        f.write(q.get(timeout=0.1))
                    except _audio_queue.Empty:
                        continue

        self._writer_thr = threading.Thread(target=_writer, name="audio-writer", daemon=True)
        self._writer_thr.start()

        ckpt_path, interval, recorder = self._ckpt_path, self.CHECKPOINT_INTERVAL_S, self

        def _checkpointer():
            with open(ckpt_path, "w", newline="", buffering=1) as cf:
                cw = csv.writer(cf)
                cw.writerow(["sample_index", "wall_time_s"])
                cf.flush()
                while not stop_flag.is_set():
                    time.sleep(interval)
                    if stop_flag.is_set(): break
                    cw.writerow([recorder._sample_count, time.time()])
                    cf.flush()

        self._ckpt_thr = threading.Thread(target=_checkpointer, name="audio-ckpt", daemon=True)
        self._ckpt_thr.start()

        def _cb(indata, frames, time_info, status):
            self._sample_count += frames
            try:
                q.put_nowait(indata.copy())
            except _audio_queue.Full:
                pass

        self._start_epoch = time.time()
        with open(self._sync_path, "w") as f:
            json.dump({"audio_start_epoch": self._start_epoch, "sample_rate": sr,
                       "num_channels": in_ch,
                       "note": "t = audio_start_epoch + i / sample_rate"}, f, indent=2)

        self._stream = sd.InputStream(device=in_dev_idx, samplerate=sr,
                                       blocksize=self._blocksize, dtype="float32",
                                       channels=in_ch, callback=_cb)
        self._stream.start()
        print(f"    Audio: {self._wav_path}  ({in_ch} ch @ {sr} Hz)")

    def stop(self):
        if self._stream:
            self._stream.stop(); self._stream.close(); self._stream = None
        self._stop_flag.set()
        if self._ckpt_thr:   self._ckpt_thr.join(timeout=3.0)
        if self._writer_thr: self._writer_thr.join(timeout=10.0)
        dur = self._sample_count / self._sr if self._sr > 0 else 0.0
        print(f"    Audio: stopped ({self._sample_count:,} samples ≈ {dur:.1f} s)")


# ─────────────────────────────────────────────────────────────────────────────
# Data logger
# ─────────────────────────────────────────────────────────────────────────────

RECORD_COLS = ["timestamp_s", "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]
MOTION_COLS = ["point_x_mm", "point_z_mm", "direction_deg", "speed_mm_s",
               "edge_x_mm", "edge_z_mm", "distance_mm"]


class DataLogger:
    POLL_HZ = 50

    def __init__(self, folder):
        os.makedirs(folder, exist_ok=True)
        self._fp = open(os.path.join(folder, "robot_pose_planned.csv"),  "w", newline="", buffering=1)
        self._fa = open(os.path.join(folder, "robot_pose_recorded.csv"), "w", newline="", buffering=1)
        self._fm = open(os.path.join(folder, "motion_parameters.csv"),   "w", newline="", buffering=1)
        self._wp = csv.writer(self._fp); self._wp.writerow(RECORD_COLS); self._fp.flush()
        self._wa = csv.writer(self._fa); self._wa.writerow(RECORD_COLS); self._fa.flush()
        self._wm = csv.writer(self._fm); self._wm.writerow(MOTION_COLS); self._fm.flush()
        self._stop_evt  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._n_planned = 0
        self._n_actual  = 0
        self._lock      = threading.Lock()

    def log_planned(self, tcp):
        self._wp.writerow([time.time()] + list(tcp[:6]))
        self._fp.flush(); self._n_planned += 1

    def log_motion_params(self, plan: MotionPlan):
        self._wm.writerow([plan.px, plan.pz, plan.theta_deg, plan.speed_mm_s,
                           plan.edge_x, plan.edge_z, plan.distance_mm])
        self._fm.flush()

    def start_polling(self, arm):
        if arm is None: return
        def _poll():
            interval = 1.0 / self.POLL_HZ
            while not self._stop_evt.is_set():
                t0 = time.time()
                ret, pos = arm.get_position()
                ts = time.time()
                if ret == 0 and pos and len(pos) >= 6:
                    with self._lock:
                        self._wa.writerow([ts] + list(pos[:6]))
                        self._fa.flush(); self._n_actual += 1
                time.sleep(max(0.0, interval - (time.time() - t0)))
        self._thread = threading.Thread(target=_poll, name="pose-poller", daemon=True)
        self._thread.start()

    def stop_polling(self):
        self._stop_evt.set()
        if self._thread: self._thread.join(timeout=2.0)

    def close(self):
        for f in (self._fp, self._fm):
            f.flush(); f.close()
        with self._lock:
            self._fa.flush(); self._fa.close()

    def counts(self):
        return self._n_planned, self._n_actual


# ─────────────────────────────────────────────────────────────────────────────
# Move primitives
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RobotState:
    t_start: float = field(default_factory=time.time)
    moves:   int   = 0

    @property
    def elapsed(self):
        return time.time() - self.t_start


def build_tcp(dx_mm, dz_mm, ox_mm, oy_mm, oz_mm, roll, pitch, yaw):
    return [ox_mm + dx_mm, oy_mm, oz_mm + dz_mm, roll, pitch, yaw]


def do_move(arm, tcp, speed, mvacc, state: RobotState, dry_run,
            logger: Optional[DataLogger] = None):
    if logger:
        logger.log_planned(tcp)
    if dry_run:
        time.sleep(0.005); state.moves += 1; return True
    ret = arm.set_position(*tcp, speed=speed, mvacc=mvacc, wait=True)
    state.moves += 1
    if ret != 0:
        print(f"\n  ⚠ set_position returned code {ret}", file=sys.stderr)
        return False
    return True


def run_radial_motion(arm, state: RobotState, plan: MotionPlan,
                      ox_mm, oy_mm, oz_mm, roll, pitch, yaw,
                      mvacc, dwell_s, dry_run,
                      logger: Optional[DataLogger] = None) -> bool:
    """
    Execute one radial stroke:
      1. (Already at starting point – caller ensures this)
      2. Move outward to edge
      3. Dwell briefly
      4. Return to starting point
    """
    if logger:
        logger.log_motion_params(plan)

    speed = plan.speed_mm_s

    # Outward move
    tcp_edge = build_tcp(plan.edge_x, plan.edge_z, ox_mm, oy_mm, oz_mm, roll, pitch, yaw)
    ok = do_move(arm, tcp_edge, speed, mvacc, state, dry_run, logger)
    if not ok: return False

    # Brief dwell at edge
    time.sleep(dwell_s)

    # Return to starting point
    tcp_start = build_tcp(plan.px, plan.pz, ox_mm, oy_mm, oz_mm, roll, pitch, yaw)
    ok = do_move(arm, tcp_start, speed, mvacc, state, dry_run, logger)
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Radial sweep data collection for XArm fingerpad",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── geometry ───────────────────────────────────────────────────────────
    parser.add_argument("--height-cm",   type=float, default=3.1,
                        help="Sensor rectangle height (Y-axis on sensor) [cm]")
    parser.add_argument("--width-cm",    type=float, default=2.5,
                        help="Sensor rectangle width [cm]")
    parser.add_argument("--origin-x-mm", type=float, default=280.5,
                        help="TCP X of sensor centre [mm]")
    parser.add_argument("--origin-z-mm", type=float, default=83.0,
                        help="TCP Z of sensor centre [mm]")

    # ── TCP fixed values ───────────────────────────────────────────────────
    parser.add_argument("--y",     type=float, default=250.5)
    parser.add_argument("--roll",  type=float, default=178)
    parser.add_argument("--pitch", type=float, default=-2)
    parser.add_argument("--yaw",   type=float, default=2)
    #TODO: Hand tune --origin-x-mm, --origin-z-mm, --y, --roll, --pitch, --yaw to get the best alignment with the calibrated object.

    # ── grid & motion params ───────────────────────────────────────────────
    parser.add_argument("--grid-mm", type=float, default=DEFAULT_GRID_MM,
                        help="Grid spacing between starting points [mm]")
    parser.add_argument("--vmin", type=float, default=DEFAULT_VMIN_MM_S,
                        help="Minimum random stroke speed [mm/s]")
    parser.add_argument("--vmax", type=float, default=DEFAULT_VMAX_MM_S,
                        help="Maximum random stroke speed [mm/s]")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help="Safety margin (fraction of half-width/height) "
                             "shrunk from all four sides")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible direction/speed sampling")

    # ── robot connection ───────────────────────────────────────────────────
    parser.add_argument("--ip",      type=str, default="192.168.1.200")
    parser.add_argument("--dry-run", action="store_true")


    # ── motion tuning ──────────────────────────────────────────────────────
    parser.add_argument("--mvacc", type=float, default=DEFAULT_MVACC_MM_S2,
                        help="Linear acceleration [mm/s²]")
    parser.add_argument("--dwell", type=float, default=DEFAULT_DWELL_S,
                        help="Dwell time at edge before returning [s]")

    # ── output ─────────────────────────────────────────────────────────────
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Base output folder; pt####_dir### subfolders created inside")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction,
                        default=True)

    # ── audio ──────────────────────────────────────────────────────────────
    parser.add_argument("--audio",       action="store_true")
    parser.add_argument("--sr",          type=int, default=48000)
    parser.add_argument("--blocksize",   type=int, default=2048)
    parser.add_argument("--device-hint", type=str, default="XR18")

    args = parser.parse_args()

    if args.vmin >= args.vmax:
        parser.error("--vmin must be less than --vmax")
    if args.mvacc > XARM_MAX_ACC_MM_S2:
        parser.error(f"--mvacc exceeds {XARM_MAX_ACC_MM_S2}")
    if not args.dry_run and not XARM_AVAILABLE:
        parser.error("xArm-Python-SDK not found.  Add --dry-run to test.")
    if args.audio and not AUDIO_AVAILABLE:
        parser.error("sounddevice/soundfile not found.  Omit --audio.")

    hw_mm = (args.width_cm  / 2.0) * 10.0 * (1.0 - args.margin)
    hh_mm = (args.height_cm / 2.0) * 10.0 * (1.0 - args.margin)

    # ── plan all motions ───────────────────────────────────────────────────
    print("  Planning trajectories ...", end="", flush=True)
    plans = plan_all(hw_mm, hh_mm, args.grid_mm, args.vmin, args.vmax, args.seed)
    n_pts  = len(set((p.pt_idx for p in plans)))
    total  = len(plans)
    print(f" ✓  ({n_pts} grid points, {total} radial motions)")

    # ── print session header ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  COLLECT RADIAL DATA")
    print(f"  Rectangle : {2*hw_mm:.1f} × {2*hh_mm:.1f} mm  (after margin)")
    print(f"  Grid      : {args.grid_mm:.1f} mm spacing  →  {n_pts} starting points")
    print(f"  Speed     : [{args.vmin:.0f}, {args.vmax:.0f}] mm/s  (random per motion)")
    print(f"  Motions   : {total}  (≈ {total/n_pts:.1f} per point on average)")
    print(f"  Seed      : {args.seed}")
    print(f"  Output    : {args.output}/pt####_dir###/")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  {'DRY-RUN' if args.dry_run else 'IP: ' + args.ip}")
    print(f"  Audio     : {'enabled @ ' + str(args.sr) + ' Hz' if args.audio else 'disabled'}")
    print(f"{SEP}\n")

    # ── connect once ───────────────────────────────────────────────────────
    arm = None
    if not args.dry_run:
        print(f"  Connecting to XArm at {args.ip} ...", end="", flush=True)
        arm = XArmAPI(args.ip)
        arm.motion_enable(enable=True)
        arm.set_mode(0); arm.set_state(state=0)
        time.sleep(0.5)
        print(" ✓")
        home_tcp = build_tcp(0, 0, args.origin_x_mm, args.y,
                              args.origin_z_mm, args.roll, args.pitch, args.yaw)
        print("  Moving to centre origin ...", end="", flush=True)
        arm.set_position(*home_tcp, speed=args.vmin, mvacc=args.mvacc, wait=True)
        print(" ✓\n")
    else:
        print("  [DRY-RUN] No robot connection.\n")

    # ── graceful Ctrl+C ────────────────────────────────────────────────────
    _stop = {"flag": False}
    def _sigint(sig, frame):
        print("\n\n  ⚡ Interrupt – finishing current motion then stopping ...")
        _stop["flag"] = True
    signal.signal(signal.SIGINT, _sigint)

    # ── main loop ──────────────────────────────────────────────────────────
    done = skipped = failed = 0
    state = RobotState()
    cur_pt_idx = None   # track current grid point to avoid redundant moves

    for motion_idx, plan in enumerate(plans):
        if _stop["flag"]:
            break

        run_folder = os.path.join(args.output, plan.folder)
        tag = f"[{motion_idx+1:>5}/{total}]  {plan.folder}  " \
              f"θ={plan.theta_deg:6.1f}°  v={plan.speed_mm_s:5.1f} mm/s  " \
              f"d={plan.distance_mm:5.1f} mm"

        if args.skip_existing and os.path.isdir(run_folder):
            print(f"  {tag}  → SKIP")
            skipped += 1
            continue

        # Move to starting point if we're on a different grid point
        if cur_pt_idx != plan.pt_idx:
            tcp_start = build_tcp(plan.px, plan.pz,
                                   args.origin_x_mm, args.y, args.origin_z_mm,
                                   args.roll, args.pitch, args.yaw)
            if arm is not None:
                arm.set_position(*tcp_start, speed=args.vmin,
                                  mvacc=args.mvacc, wait=True)
            elif args.dry_run:
                time.sleep(0.01)
            cur_pt_idx = plan.pt_idx

        print(f"  {tag}")

        logger = DataLogger(run_folder)
        if arm: logger.start_polling(arm)

        audio_rec = None
        if args.audio:
            audio_rec = AudioRecorder(run_folder, sr=args.sr,
                                       blocksize=args.blocksize,
                                       device_hint=args.device_hint)
            audio_rec.start()

        ok = run_radial_motion(
            arm=arm, state=state, plan=plan,
            ox_mm=args.origin_x_mm, oy_mm=args.y, oz_mm=args.origin_z_mm,
            roll=args.roll, pitch=args.pitch, yaw=args.yaw,
            mvacc=args.mvacc, dwell_s=args.dwell, dry_run=args.dry_run,
            logger=logger,
        )

        if audio_rec: audio_rec.stop()
        logger.stop_polling(); logger.close()

        if ok:
            done += 1
        else:
            failed += 1
            print(f"\n  ⚠ Arm fault at {plan.folder} – aborting session.", file=sys.stderr)
            _stop["flag"] = True

    # ── session summary ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  Session complete")
    print(f"  Done: {done}  |  Skipped: {skipped}  |  Failed: {failed}  |  Total: {total}")

    if arm:
        print("  Returning to origin ...", end="", flush=True)
        home_tcp = build_tcp(0, 0, args.origin_x_mm, args.y,
                              args.origin_z_mm, args.roll, args.pitch, args.yaw)
        arm.set_position(*home_tcp, speed=args.vmin, mvacc=args.mvacc, wait=True)
        print(" ✓")
        arm.disconnect()
        print("  Disconnected.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()