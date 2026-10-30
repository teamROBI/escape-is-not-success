# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument('--suite', required=True)
_parser.add_argument('--demos', type=int, default=5, help='successful demonstrations required per condition')
_parser.add_argument('--splits', default='train', help='comma-separated splits to collect: train, interior, test')
_parser.add_argument('--partition-idx', type=int, default=0)
_parser.add_argument('--partition-total', type=int, default=1)
_parser.add_argument('--max-attempts', type=int, default=3, help='attempts per required demonstration before giving up on the suite')
_parser.add_argument('--max-steps', type=int, default=3000)
_parser.add_argument('--depth', action='store_true', help='also record metric depth per camera as obs/depth_<cam> (float32 metres, ~1 MB/frame/camera -- opt-in for that reason)')
_parser.add_argument('--no-images', action='store_true', help='record actions and states only. For timing the loop without paying for rendering; the result is NOT trainable.')
_parser.add_argument('--no-jitter', action='store_true', help='start every episode from the exact home pose. Makes repeated demos of one condition byte-identical, so only useful with --demos 1.')
_parser.add_argument('--out', default='data/output/collect')
_parser.add_argument('--attempt-start', type=int, default=0, help='first attempt number per condition. The counter restarts in every run, so a second collection of the same cells would repeat attempts 0, 1, ... -- the SAME jitter seeds (episode_seed(key, attempt)) as the first. Set this past the attempts already collected.')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = not args_cli.no_images
_app = AppLauncher(args_cli).app
import numpy as np
from escape_ood.collect.context import ObjectDropped, PhaseTimeout, TaskContext, TaskFailed
from escape_ood.collect.recorder import EpisodeRecorder, archive_summary, next_episode_index
from escape_ood.collect.task import ConditionQueue
from escape_ood.collect.tasks import task_for
from escape_ood.sim.env.isaac_backend import IsaacMotionBackend
from escape_ood.sim.env.scene import assert_condition_realised, build_scene, episode_seed, jittered_home, reset_condition_scene, suite_home
from escape_ood.suites.spec import object_keys_for, suite as get_suite

def grab(scene, cams: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for cam in cams:
        rgb = scene[cam].data.output.get('rgb')
        if rgb is None:
            continue
        img = rgb[0, ..., :3].detach().cpu().numpy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        out[cam] = img
    return out

def grab_depth(scene, cams: list[str]) -> dict[str, np.ndarray]:
    pass
    out: dict[str, np.ndarray] = {}
    for cam in cams:
        d = scene[cam].data.output.get('distance_to_image_plane')
        if d is None:
            continue
        arr = d[0].detach().cpu().numpy()
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f'{cam}: expected an (H, W) depth map, got {arr.shape}')
        out[cam] = arr.astype(np.float32)
    return out

def phase_spans(ctx: TaskContext, discard_frames: int) -> list[list]:
    """``[[label, start, end, ok], ...]`` in RECORDED frame indices.

    ``ctx.phases`` counts every control step including the discard prologue; the recorder
    drops the first ``discard_frames``. A phase that ends inside the prologue is not in the
    archive and is omitted; one that straddles it is clipped.
    """
    (out, t) = ([], 0)
    for p in ctx.phases:
        (a, b) = (t - discard_frames, t + p.steps - discard_frames)
        t += p.steps
        if b <= 0:
            continue
        out.append([str(p.label), max(a, 0), b, bool(p.ok)])
    return out

def main() -> int:
    suite_name = args_cli.suite
    s = get_suite(suite_name)
    object_keys = object_keys_for(suite_name)
    splits = tuple((x.strip() for x in args_cli.splits.split(',') if x.strip()))
    (scene, sim, info) = build_scene(suite_name, device=args_cli.device, object_keys=object_keys, with_cameras=not args_cli.no_images)
    cams = [] if args_cli.no_images else [k for k in scene.keys() if k.startswith('cam')]
    backend = IsaacMotionBackend(scene, sim)
    queue = ConditionQueue(s, object_keys, demos_per_condition=args_cli.demos, splits=splits, partition_idx=args_cli.partition_idx, partition_total=args_cli.partition_total)
    out_dir = Path(args_cli.out)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    tag = suite_name
    if splits != ('train',):
        tag += '_' + '+'.join(splits)
    if args_cli.partition_total > 1:
        tag += f'_p{args_cli.partition_idx}of{args_cli.partition_total}'
    archive = out_dir / f'{tag}.hdf5'
    ep_index = next_episode_index(archive)
    depth_cams = cams if args_cli.depth else []
    recorder = EpisodeRecorder(image_keys=cams, depth_keys=depth_cams)
    backend.attach_recorder(recorder)
    print(f'[COLLECT] {queue.summary()}', flush=True)
    print(f"[COLLECT] cameras={cams or 'none'} depth={depth_cams or 'none'} -> {archive} (resuming at episode {ep_index})", flush=True)
    if cams:
        for _ in range(30):
            sim.render()
            scene.update(0.0)
    budget = queue.total_episodes * max(1, args_cli.max_attempts)
    attempts = n_ok = n_fail = 0
    attempt_of: dict[str, int] = {}
    t0 = time.time()
    while not queue.done and attempts < budget:
        cond = queue.next_condition()
        if cond is None:
            break
        attempts += 1
        assert_condition_realised(suite_name, cond.params, info['built'])
        rep = attempt_of.get(cond.key(), args_cli.attempt_start)
        attempt_of[cond.key()] = rep + 1
        seed = episode_seed(cond.key(), rep)
        base = suite_home(suite_name)
        home = base if args_cli.no_jitter else jittered_home(seed, base=base)
        backend.reset_to(home)
        reset_condition_scene(scene, sim, info, cond)
        ctx = TaskContext(backend)
        ctx.object_name = info['instance_name'][cond.object_key]
        task = task_for(cond)(cond)
        recorder.reset()
        (ok, reason, n) = (True, '', 0)
        try:
            trajectory = task.run(ctx)
            for _ in trajectory:
                if cams:
                    sim.render()
                    scene.update(0.0)
                pose = backend.last_pose()
                if pose is None:
                    raise RuntimeError('expert issued a joint-space command; collection requires the EEF command on every frame')
                (p, q, g) = pose
                (ep, eq) = backend.getp()
                recorder.step(action=backend.last_action(), joint_pos=backend.getj() + [backend.gripper_pos()], images=grab(scene, cams) if cams else None, depths=grab_depth(scene, depth_cams) if depth_cams else None, eef_target=list(p) + list(q) + [g], eef_pose=list(ep) + list(eq))
                backend.step()
                n += 1
                if n >= args_cli.max_steps:
                    raise TimeoutError(f'exceeded {args_cli.max_steps} control steps')
        except (PhaseTimeout, ObjectDropped, TaskFailed, TimeoutError) as e:
            (ok, reason) = (False, f'{type(e).__name__}: {e}')
            if isinstance(e, TaskFailed) and (not e.retriable):
                print(f'[COLLECT] FATAL non-retriable on {cond.key()}: {e}', flush=True)
                break
        if recorder.n_frames:
            attrs = cond.to_attrs()
            attrs['episode_seed'] = int(seed)
            attrs['attempt'] = int(rep)
            attrs['reset_home_joint_pos'] = json.dumps(home)
            attrs['phases'] = json.dumps(phase_spans(ctx, len(recorder.actions) - recorder.n_frames))
            recorder.write(archive, ep_index, attrs, success=ok)
            ep_index += 1
        if ok:
            n_ok += 1
            queue.confirm_success(cond)
        else:
            n_fail += 1
            queue.record_failure(cond)
        if attempts % 10 == 0 or not ok:
            rate = (time.time() - t0) / max(attempts, 1)
            left = (queue.total_episodes - queue.collected) * rate
            print(f'[COLLECT] {queue.collected}/{queue.total_episodes} kept ({n_fail} failed) {rate:.1f}s/ep  eta {left / 60:.0f}m' + (f'  last-fail {cond.key()}: {reason}' if not ok else ''), flush=True)
    dt = time.time() - t0
    done = queue.done
    print(f"[COLLECT] {('COMPLETE' if done else 'INCOMPLETE')} {queue.collected}/{queue.total_episodes} in {dt / 60:.1f}m ({n_ok} ok, {n_fail} failed, {attempts} attempts)", flush=True)
    print(f'[COLLECT] archive {archive}: {archive_summary(archive)}', flush=True)
    sys.stdout.flush()
    import os
    os._exit(0 if done else 1)

def _hard_exit(code: int) -> None:
    """Terminate for real.

    Isaac's app holds ~200 non-daemon threads, so a bare ``raise`` or ``SystemExit`` unwinds the
    stack and then hangs the process indefinitely instead of exiting. Every exit path after
    ``AppLauncher`` therefore goes through ``os._exit``.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(code)
if __name__ == '__main__':
    try:
        main()
    except SystemExit as e:
        _hard_exit(int(e.code) if isinstance(e.code, int) else 1)
    except BaseException:
        import traceback
        traceback.print_exc()
        _hard_exit(1)
