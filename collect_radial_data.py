#!/usr/bin/env python3
"""
collect_radial_data.py  –  Pitch-rotation data collection for XArm fingerpad

Generates random rotation paths in the pitch direction.  For each path:
  - A rotation centre is chosen within a ~1"x2" box around the gripper
    endpoint by setting the xArm TCP offset
  - Random start and end pitch angles are chosen within [--pitch-min … --pitch-max]
  - A random speed is chosen for the motion

The arm rotates from start_pitch to end_pitch, dwells briefly, then returns.

For rotation-only moves the xArm interprets the speed parameter as rotational
speed (0-1000 ≈ 0-180 deg/s).

Folder layout
─────────────
  {output}/
    path{NNNN}/
        robot_pose_planned.csv
        robot_pose_recorded.csv
        motion_parameters.csv   (offset_x_mm, offset_z_mm, start_pitch_deg,
                                  end_pitch_deg, speed)
        audio.wav               (only with --audio)
        audio_sync.json
        audio_timestamps.csv

Usage
─────
  # Dry-run preview:
  python collect_radial_data.py -o ./data --dry-run

  # Live robot:
  python collect_radial_data.py -o ./data --ip 192.168.1.200 --audio

  # Custom parameters:
  python collect_radial_data.py -o ./data --num-paths 200 --vmin 20 --vmax 120
"""

import argparse
import csv
import json
import os
import queue as _audio_queue
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

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
DEFAULT_NUM_PATHS    = 500
DEFAULT_VMIN         = 10.0
DEFAULT_VMAX         = 50.0
DEFAULT_MVACC_MM_S2  = 200.0
DEFAULT_DWELL_S      = 0.3
DEFAULT_PITCH_MIN    = -100.0
DEFAULT_PITCH_MAX    = 100.0
OFFSET_BOX_X_MM      = 25.4    # 1 inch
OFFSET_BOX_Z_MM      = 50.8    # 2 inches

SEP = "─" * 64


# ─────────────────────────────────────────────────────────────────────────────
# Plan all motions up-front
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MotionPlan:
    path_idx:        int
    offset_x_mm:     float   # TCP offset X (rotation centre, tool frame)
    offset_z_mm:     float   # TCP offset Z (rotation centre, tool frame)
    start_pitch_deg: float   # starting pitch angle
    end_pitch_deg:   float   # ending pitch angle
    speed:           float   # motion speed (xArm speed parameter)
    folder:          str


def plan_all(num_paths: int, vmin: float, vmax: float,
             pitch_min: float, pitch_max: float,
             seed: int = 42) -> list[MotionPlan]:
    rng = random.Random(seed)
    plans: list[MotionPlan] = []
    half_x = OFFSET_BOX_X_MM / 2.0   # ±12.7 mm
    half_z = OFFSET_BOX_Z_MM / 2.0   # ±25.4 mm

    for i in range(num_paths):
        offset_x = rng.uniform(-half_x, half_x)
        offset_z = rng.uniform(-half_z, half_z)
        start_pitch = rng.uniform(pitch_min, pitch_max)
        end_pitch   = rng.uniform(pitch_min, pitch_max)
        # Ensure start and end differ by at least 5°
        while abs(end_pitch - start_pitch) < 5.0:
            end_pitch = rng.uniform(pitch_min, pitch_max)
        speed  = rng.uniform(vmin, vmax)
        folder = f"path{i:04d}"
        plans.append(MotionPlan(
            path_idx=i,
            offset_x_mm=offset_x,
            offset_z_mm=offset_z,
            start_pitch_deg=start_pitch,
            end_pitch_deg=end_pitch,
            speed=speed,
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
MOTION_COLS = ["offset_x_mm", "offset_z_mm", "start_pitch_deg", "end_pitch_deg", "speed"]


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
        self._wm.writerow([plan.offset_x_mm, plan.offset_z_mm,
                           plan.start_pitch_deg, plan.end_pitch_deg,
                           plan.speed])
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


def run_rotation_motion(arm, state: RobotState, plan: MotionPlan,
                        ox_mm, oy_mm, oz_mm, roll, yaw,
                        base_tcp_offset, mvacc, dwell_s, dry_run,
                        logger: Optional[DataLogger] = None) -> bool:
    """
    Execute one rotation path:
      1. Set TCP offset to define the rotation centre
      2. Move to start pitch
      3. Rotate to end pitch
      4. Dwell briefly
      5. Return to start pitch
      6. Restore original TCP offset
    """
    if logger:
        logger.log_motion_params(plan)

    speed = plan.speed

    # Set TCP offset: base (saved) offset + path-specific rotation centre
    if arm is not None:
        path_offset = [base_tcp_offset[0] + plan.offset_x_mm,
                       base_tcp_offset[1],
                       base_tcp_offset[2] + plan.offset_z_mm,
                       base_tcp_offset[3],
                       base_tcp_offset[4],
                       base_tcp_offset[5]]
        arm.set_tcp_offset(path_offset)
        time.sleep(0.5)

    # TCP positions: same XYZ, only pitch changes
    tcp_start = [ox_mm, oy_mm, oz_mm, roll, plan.start_pitch_deg, yaw]
    tcp_end   = [ox_mm, oy_mm, oz_mm, roll, plan.end_pitch_deg,   yaw]

    # Move to start pitch
    ok = do_move(arm, tcp_start, speed, mvacc, state, dry_run, logger)
    if not ok: return False

    # Rotate to end pitch
    ok = do_move(arm, tcp_end, speed, mvacc, state, dry_run, logger)
    if not ok: return False

    # Dwell at end position
    time.sleep(dwell_s)

    # Return to start pitch
    ok = do_move(arm, tcp_start, speed, mvacc, state, dry_run, logger)
    if not ok: return False

    # Restore original TCP offset
    if arm is not None:
        arm.set_tcp_offset(base_tcp_offset)
        time.sleep(0.2)

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pitch-rotation data collection for XArm fingerpad",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── geometry ───────────────────────────────────────────────────────────
    parser.add_argument("--origin-x-mm", type=float, default=280.5,
                        help="TCP X of rotation reference point [mm]")
    parser.add_argument("--origin-z-mm", type=float, default=83.0,
                        help="TCP Z of rotation reference point [mm]")

    # ── TCP fixed values ───────────────────────────────────────────────────
    parser.add_argument("--y",     type=float, default=250.5)
    parser.add_argument("--roll",  type=float, default=178)
    parser.add_argument("--yaw",   type=float, default=2)
    #TODO: Hand tune --origin-x-mm, --origin-z-mm, --y, --roll, --yaw to get the best alignment with the calibrated object.

    # ── rotation params ───────────────────────────────────────────────────
    parser.add_argument("--num-paths", type=int, default=DEFAULT_NUM_PATHS,
                        help="Number of rotation paths to generate")
    parser.add_argument("--pitch-min", type=float, default=DEFAULT_PITCH_MIN,
                        help="Minimum pitch angle [deg]")
    parser.add_argument("--pitch-max", type=float, default=DEFAULT_PITCH_MAX,
                        help="Maximum pitch angle [deg]")
    parser.add_argument("--vmin", type=float, default=DEFAULT_VMIN,
                        help="Minimum random speed (xArm speed parameter)")
    parser.add_argument("--vmax", type=float, default=DEFAULT_VMAX,
                        help="Maximum random speed (xArm speed parameter)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")

    # ── robot connection ───────────────────────────────────────────────────
    parser.add_argument("--ip",      type=str, default="192.168.1.200")
    parser.add_argument("--dry-run", action="store_true")

    # ── motion tuning ──────────────────────────────────────────────────────
    parser.add_argument("--mvacc", type=float, default=DEFAULT_MVACC_MM_S2,
                        help="Acceleration [mm/s²]")
    parser.add_argument("--dwell", type=float, default=DEFAULT_DWELL_S,
                        help="Dwell time at end pitch before returning [s]")

    # ── output ─────────────────────────────────────────────────────────────
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Base output folder; path#### subfolders created inside")
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
    if args.pitch_min >= args.pitch_max:
        parser.error("--pitch-min must be less than --pitch-max")
    if args.mvacc > XARM_MAX_ACC_MM_S2:
        parser.error(f"--mvacc exceeds {XARM_MAX_ACC_MM_S2}")
    if not args.dry_run and not XARM_AVAILABLE:
        parser.error("xArm-Python-SDK not found.  Add --dry-run to test.")
    if args.audio and not AUDIO_AVAILABLE:
        parser.error("sounddevice/soundfile not found.  Omit --audio.")

    # ── plan all motions ───────────────────────────────────────────────────
    print("  Planning rotation paths ...", end="", flush=True)
    plans = plan_all(args.num_paths, args.vmin, args.vmax,
                     args.pitch_min, args.pitch_max, args.seed)
    total = len(plans)
    print(f" ✓  ({total} rotation paths)")

    # ── print session header ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  COLLECT ROTATION DATA")
    print(f"  Pitch range : [{args.pitch_min:.0f}°, {args.pitch_max:.0f}°]")
    print(f"  Offset box  : {OFFSET_BOX_X_MM:.1f} × {OFFSET_BOX_Z_MM:.1f} mm  (1\" × 2\")")
    print(f"  Speed       : [{args.vmin:.0f}, {args.vmax:.0f}]  (random per path)")
    print(f"  Paths       : {total}")
    print(f"  Seed        : {args.seed}")
    print(f"  Output      : {args.output}/path####/")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  {'DRY-RUN' if args.dry_run else 'IP: ' + args.ip}")
    print(f"  Audio       : {'enabled @ ' + str(args.sr) + ' Hz' if args.audio else 'disabled'}")
    print(f"{SEP}\n")

    # ── connect once ───────────────────────────────────────────────────────
    arm = None
    base_tcp_offset = [0, 0, 0, 0, 0, 0]
    if not args.dry_run:
        print(f"  Connecting to XArm at {args.ip} ...", end="", flush=True)
        arm = XArmAPI(args.ip)
        arm.motion_enable(enable=True)
        arm.set_mode(0); arm.set_state(state=0)
        time.sleep(0.5)
        # Save the arm's current TCP offset so we can restore it later
        base_tcp_offset = list(arm.tcp_offset)
        print(" ✓")
        # Move to reference position with neutral pitch (0°)
        home_tcp = [args.origin_x_mm, args.y, args.origin_z_mm,
                    args.roll, 0, args.yaw]
        print("  Moving to reference position ...", end="", flush=True)
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

    for path_idx, plan in enumerate(plans):
        if _stop["flag"]:
            break

        run_folder = os.path.join(args.output, plan.folder)
        tag = f"[{path_idx+1:>5}/{total}]  {plan.folder}  " \
              f"offset=({plan.offset_x_mm:+6.1f}, {plan.offset_z_mm:+6.1f})  " \
              f"pitch={plan.start_pitch_deg:+6.1f}→{plan.end_pitch_deg:+6.1f}°  " \
              f"v={plan.speed:5.1f}"

        if args.skip_existing and os.path.isdir(run_folder):
            print(f"  {tag}  → SKIP")
            skipped += 1
            continue

        print(f"  {tag}")

        logger = DataLogger(run_folder)
        if arm: logger.start_polling(arm)

        audio_rec = None
        if args.audio:
            audio_rec = AudioRecorder(run_folder, sr=args.sr,
                                       blocksize=args.blocksize,
                                       device_hint=args.device_hint)
            audio_rec.start()

        ok = run_rotation_motion(
            arm=arm, state=state, plan=plan,
            ox_mm=args.origin_x_mm, oy_mm=args.y, oz_mm=args.origin_z_mm,
            roll=args.roll, yaw=args.yaw,
            base_tcp_offset=base_tcp_offset,
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
        # Restore original TCP offset and return to reference position
        arm.set_tcp_offset(base_tcp_offset)
        time.sleep(0.5)
        print("  Returning to reference position ...", end="", flush=True)
        home_tcp = [args.origin_x_mm, args.y, args.origin_z_mm,
                    args.roll, 0, args.yaw]
        arm.set_position(*home_tcp, speed=args.vmin, mvacc=args.mvacc, wait=True)
        print(" ✓")
        arm.disconnect()
        print("  Disconnected.")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
