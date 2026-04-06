#!/usr/bin/env python3
"""
test_radial.py  –  Test radial data collection for a single point + direction.

Picks one grid point (by index, or random) and one random direction/speed,
then executes the outward + return stroke.  Useful for verifying robot motion,
audio sync, and file output before a full session.

Usage
─────
  # Random point, random direction, dry-run:
  python test_radial.py --dry-run

  # Specific grid point index, dry-run:
  python test_radial.py --point 42 --dry-run

  # Live robot, random point:
  python test_radial.py --ip 192.168.1.200

  # Live robot, specific point, with audio:
  python test_radial.py --ip 192.168.1.200 --point 42 --audio

  # Override speed (ignore random):
  python test_radial.py --dry-run --speed 25.0
"""

import argparse
import random
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("NumPy required:  pip install numpy")

# ── import from the collection script ──────────────────────────────────────
import importlib.util, os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "collect_radial_data",
    os.path.join(_here, "collect_radial_data.py"))
_crd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_crd)

plan_all          = _crd.plan_all
build_grid        = _crd.build_grid
compute_valid_arc = _crd.compute_valid_arc
sample_directions = _crd.sample_directions
ray_to_edge       = _crd.ray_to_edge
run_radial_motion = _crd.run_radial_motion
build_tcp         = _crd.build_tcp
DataLogger        = _crd.DataLogger
AudioRecorder     = _crd.AudioRecorder
RobotState        = _crd.RobotState
MotionPlan        = _crd.MotionPlan
DEFAULT_GRID_MM   = _crd.DEFAULT_GRID_MM
DEFAULT_VMIN_MM_S = _crd.DEFAULT_VMIN_MM_S
DEFAULT_VMAX_MM_S = _crd.DEFAULT_VMAX_MM_S
DEFAULT_MARGIN    = _crd.DEFAULT_MARGIN
XARM_AVAILABLE    = _crd.XARM_AVAILABLE
AUDIO_AVAILABLE   = _crd.AUDIO_AVAILABLE

SEP = "─" * 64


def main():
    parser = argparse.ArgumentParser(
        description="Randomized radial motion test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── geometry (must match your collection params) ───────────────────────
    parser.add_argument("--height-cm",   type=float, default=3.1)
    parser.add_argument("--width-cm",    type=float, default=2.5)
    parser.add_argument("--grid-mm",     type=float, default=DEFAULT_GRID_MM)
    parser.add_argument("--margin",      type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--origin-x-mm", type=float, default=280.5)
    parser.add_argument("--origin-z-mm", type=float, default=83.0)
    parser.add_argument("--y",       type=float, default=250.5)
    parser.add_argument("--roll",    type=float, default=178)
    parser.add_argument("--pitch",   type=float, default=-2)
    parser.add_argument("--yaw",     type=float, default=2)

    #TODO: Hand tune --origin-x-mm, --origin-z-mm, --y, --roll, --pitch, --yaw to get the best alignment with the calibrated object.

    # ── test selection ─────────────────────────────────────────────────────
    parser.add_argument("--point", type=int, default=None,
                        help="Grid point index to test (default: random)")
    parser.add_argument("--speed", type=float, default=None,
                        help="Override speed [mm/s] (default: random in [vmin, vmax])")
    parser.add_argument("--vmin",  type=float, default=DEFAULT_VMIN_MM_S)
    parser.add_argument("--vmax",  type=float, default=DEFAULT_VMAX_MM_S)
    parser.add_argument("--seed",  type=int, default=None,
                        help="Random seed (default: truly random)")
    parser.add_argument("-n", "--cycles", type=int, default=1,
                        help="Number of randomized motion cycles to run")

    # ── robot ──────────────────────────────────────────────────────────────
    parser.add_argument("--ip",      type=str, default="192.168.1.200")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mvacc",   type=float, default=200.0)
    parser.add_argument("--dwell",   type=float, default=0.3)

    # ── output & audio ─────────────────────────────────────────────────────
    parser.add_argument("-o", "--output", type=str, default="./test_output",
                        help="Folder to save test data")
    parser.add_argument("--audio",        action="store_true")
    parser.add_argument("--sr",           type=int, default=48000)
    parser.add_argument("--blocksize",    type=int, default=2048)
    parser.add_argument("--device-hint",  type=str, default="XR18")

    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be >= 1")

    if not args.dry_run and not XARM_AVAILABLE:
        parser.error("xArm-Python-SDK not found.  Add --dry-run to test.")
    if args.audio and not AUDIO_AVAILABLE:
        parser.error("sounddevice/soundfile not found.  Omit --audio.")

    rng = random.Random(args.seed)

    hw_mm = (args.width_cm  / 2.0) * 10.0 * (1.0 - args.margin)
    hh_mm = (args.height_cm / 2.0) * 10.0 * (1.0 - args.margin)

    grid = build_grid(hw_mm, hh_mm, args.grid_mm)
    n_pts = len(grid)
    if args.point is not None and not (0 <= args.point < n_pts):
        sys.exit(f"  ✗ --point {args.point} out of range [0, {n_pts-1}]")

    # ── print test plan ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  RADIAL MOTION TEST")
    print(f"  Cycles     : {args.cycles}")
    print(f"  Grid point : "
          f"{'#' + str(args.point) if args.point is not None else 'random each cycle'}")
    print(f"  Direction  : random each cycle")
    print(f"  Speed      : "
          f"{args.speed:.1f} mm/s [overridden]" if args.speed is not None
          else f"random in [{args.vmin:.1f}, {args.vmax:.1f}] mm/s")
    print(f"  Seed       : {args.seed}")
    print(f"  Output     : {args.output}")
    print(f"  {'DRY-RUN' if args.dry_run else 'LIVE – IP: ' + args.ip}")
    print(f"  Audio      : {'enabled' if args.audio else 'disabled'}")
    print(f"{SEP}\n")

    # ── connect ────────────────────────────────────────────────────────────
    arm = None
    if not args.dry_run:
        from xarm.wrapper import XArmAPI
        print(f"  Connecting to {args.ip} ...", end="", flush=True)
        arm = XArmAPI(args.ip)
        arm.motion_enable(enable=True)
        arm.set_mode(0); arm.set_state(state=0)
        import time; time.sleep(0.5)
        print(" ✓")

        print("  Connected.\n")
    else:
        print("  [DRY-RUN] Skipping robot connection.\n")

    # ── execute ────────────────────────────────────────────────────────────
    state = RobotState()
    ok_all = True
    total_planned = 0
    total_actual = 0

    for cycle_idx in range(args.cycles):
        pt_idx = args.point if args.point is not None else rng.randrange(n_pts)
        px, pz = grid[pt_idx]

        lo_deg, arc_deg = compute_valid_arc(px, pz, hw_mm, hh_mm)
        theta_deg = sample_directions(lo_deg, arc_deg, 1, rng)[0]
        speed = args.speed if args.speed is not None else rng.uniform(args.vmin, args.vmax)
        ex, ez = ray_to_edge(px, pz, theta_deg, hw_mm, hh_mm)
        distance = float(np.hypot(ex - px, ez - pz))

        folder_name = f"cycle{cycle_idx+1:03d}_pt{pt_idx:04d}_dir000"
        run_folder = os.path.join(args.output, folder_name)
        plan = MotionPlan(
            pt_idx=pt_idx, dir_idx=0,
            px=px, pz=pz,
            theta_deg=theta_deg,
            speed_mm_s=speed,
            edge_x=ex, edge_z=ez,
            distance_mm=distance,
            folder=folder_name,
        )

        print(f"\n{SEP}")
        print(f"  Cycle {cycle_idx+1}/{args.cycles}")
        print(f"  Grid point : #{pt_idx} of {n_pts}  →  ({px:.2f}, {pz:.2f}) mm")
        print(f"  Direction  : {theta_deg:.1f}°   (valid arc: {lo_deg:.1f}° + {arc_deg:.1f}°)")
        print(f"  Edge point : ({ex:.2f}, {ez:.2f}) mm")
        print(f"  Distance   : {distance:.2f} mm  (×2 round-trip = {2*distance:.2f} mm)")
        print(f"  Speed      : {speed:.1f} mm/s"
              + ("  [overridden]" if args.speed is not None else "  [random]"))
        print(f"  Output     : {run_folder}")
        print(f"{SEP}")

        if arm:
            print(f"  Moving to starting point ({px:.2f}, {pz:.2f}) mm ...", end="", flush=True)
            tcp_start = build_tcp(px, pz, args.origin_x_mm, args.y,
                                  args.origin_z_mm, args.roll, args.pitch, args.yaw)
            arm.set_position(*tcp_start, speed=args.vmin, mvacc=args.mvacc, wait=True)
            print(" ✓")

        logger = DataLogger(run_folder)
        if arm:
            logger.start_polling(arm)

        audio_rec = None
        if args.audio:
            audio_rec = AudioRecorder(run_folder, sr=args.sr,
                                      blocksize=args.blocksize,
                                      device_hint=args.device_hint)
            audio_rec.start()

        print(f"  ► Outward stroke  ({px:.2f},{pz:.2f}) → ({ex:.2f},{ez:.2f})  "
              f"@ {speed:.1f} mm/s ...")
        ok = run_radial_motion(
            arm=arm, state=state, plan=plan,
            ox_mm=args.origin_x_mm, oy_mm=args.y, oz_mm=args.origin_z_mm,
            roll=args.roll, pitch=args.pitch, yaw=args.yaw,
            mvacc=args.mvacc, dwell_s=args.dwell, dry_run=args.dry_run,
            logger=logger,
        )
        print("  ◄ Return stroke complete")

        if audio_rec:
            audio_rec.stop()
        logger.stop_polling()
        logger.close()
        n_planned, n_actual = logger.counts()
        total_planned += n_planned
        total_actual += n_actual

        if not ok:
            ok_all = False
            print("  ✗ Cycle failed; stopping remaining cycles.")
            break

    print(f"\n{SEP}")
    if ok_all:
        print(f"  ✓ Test passed ({args.cycles} cycle{'s' if args.cycles != 1 else ''})")
    else:
        print(f"  ✗ Test FAILED – arm returned an error")
    print(f"  Planned poses logged : {total_planned}")
    print(f"  Actual poses logged  : {total_actual}")
    print(f"  Elapsed              : {state.elapsed:.2f} s")
    print(f"  Output folder        : {args.output}/")
    print(f"{SEP}\n")

    if arm:
        print("  Returning to origin ...", end="", flush=True)
        home_tcp = build_tcp(0, 0, args.origin_x_mm, args.y,
                              args.origin_z_mm, args.roll, args.pitch, args.yaw)
        arm.set_position(*home_tcp, speed=args.vmin, mvacc=args.mvacc, wait=True)
        print(" ✓")
        arm.disconnect()

    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()