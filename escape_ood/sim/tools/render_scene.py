# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_parser.add_argument('--suite', default='lift')
_parser.add_argument('--cell', default=None, help="'r,theta_deg' of a spec cell")
_parser.add_argument('--band', default=None, help='a band label (R1..A3); first cell of it')
_parser.add_argument('--object-at', default=None, help="override the object position 'x,y' (base frame), no condition")
_parser.add_argument('--settle', type=int, default=60, help='control steps holding the home pose before capture')
_parser.add_argument('--out', default='data/output/media/renders')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(_parser)
args_cli = _parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
_app = AppLauncher(args_cli).app
import numpy as np
from escape_ood.sim.env.isaac_backend import IsaacMotionBackend
from escape_ood.sim.env.scene import build_scene, place_object, reset_condition_scene, suite_home
from escape_ood.suites.spec import SUITES, band_label, enumerate_conditions, object_keys_for

def _pick_condition(suite_name: str):
    conds = enumerate_conditions(SUITES[suite_name], object_keys_for(suite_name)[:1])
    if args_cli.cell:
        (r, th) = (float(v) for v in args_cli.cell.split(','))
        for c in conds:
            if abs(float(c.params['r']) - r) < 1e-06 and abs(float(c.params['theta_deg']) - th) < 1e-06:
                return c
        raise SystemExit(f'[RENDER] no cell r={r} theta={th} in {suite_name}')
    if args_cli.band:
        for c in conds:
            if band_label(c.params) == args_cli.band:
                return c
        raise SystemExit(f'[RENDER] no band {args_cli.band!r} in {suite_name}')
    return max(conds, key=lambda c: (c.sweep_coord, c.key()))

def save(scene, cam: str, path: Path) -> bool:
    rgb = scene[cam].data.output.get('rgb')
    if rgb is None:
        return False
    img = rgb[0, ..., :3].detach().cpu().numpy()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    try:
        import cv2
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except ImportError:
        from PIL import Image
        Image.fromarray(img).save(str(path))
    return True

def render(suite_name: str, out_dir: Path) -> list[Path]:
    (scene, sim, info) = build_scene(suite_name, device=args_cli.device, object_keys=object_keys_for(suite_name), with_cameras=True)
    backend = IsaacMotionBackend(scene, sim)
    backend.reset_to(suite_home(suite_name))
    if args_cli.object_at:
        xy = tuple((float(v) for v in args_cli.object_at.split(',')))
        place_object(scene, sim, info, info['object_keys'][0], xy)
        tag = f'{suite_name}_at{xy[0]:g}_{xy[1]:g}'
        print(f'[RENDER] object at {xy} (no condition)', flush=True)
    else:
        cond = _pick_condition(suite_name)
        reset_condition_scene(scene, sim, info, cond)
        band = band_label(cond.params) or cond.split
        tag = f"{suite_name}_{band}_r{cond.params['r']:g}_th{cond.params['theta_deg']:g}"
        print(f'[RENDER] condition: {cond.key()}  split={cond.split} band={band} d={cond.sweep_coord:.1f}cm', flush=True)
    for _ in range(args_cli.settle):
        backend.step()
    for _ in range(30):
        sim.render()
        scene.update(0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    cams = sorted((k for k in scene.keys() if k.startswith('cam')))
    written: list[Path] = []
    for cam in cams:
        p = out_dir / f"{tag}_{cam.replace('cam_', '')}.png"
        if save(scene, cam, p):
            written.append(p)
            print(f'[RENDER] wrote {p}', flush=True)
    return written

def main() -> None:
    out = Path(args_cli.out)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    written = render(args_cli.suite, out)
    print(f'[RENDER] {len(written)} image(s)', flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(0)
if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        import os
        os._exit(1)
