# Copyright 2026 ROBI Contributors.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    print('[ERROR] pxr not found. Run via: escape_ood/sim/tools/usd_python.sh escape_ood/sim/tools/create_scene.py', flush=True)
    sys.exit(1)
try:
    from pxr import PhysxSchema
    _HAS_PHYSX = True
except ImportError:
    _HAS_PHYSX = False
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from escape_ood.suites.spec import WORKSPACE
OUTPUT_DIR = _REPO_ROOT / 'data' / 'output' / 'assets' / 'scene'
STATIC_FRICTION = 1.0
DYNAMIC_FRICTION = 1.0
RESTITUTION = 0.0
TABLE_MARGIN_M = 0.1
TABLE_THICKNESS_M = 0.04
TABLE_COLOR = (0.42, 0.43, 0.46)

def _add_static_physics_material(stage, root_path, collision_prim) -> None:
    """UsdPhysics material bound to a *static* collider (no rigid body involved)."""
    mat_scope_path = root_path.AppendChild('PhysicsMaterials')
    mat_path = mat_scope_path.AppendChild('PMat_0')
    UsdGeom.Scope.Define(stage, str(mat_scope_path))
    mat = UsdShade.Material.Define(stage, str(mat_path))
    mat_prim = mat.GetPrim()
    pmat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    pmat.CreateStaticFrictionAttr().Set(STATIC_FRICTION)
    pmat.CreateDynamicFrictionAttr().Set(DYNAMIC_FRICTION)
    pmat.CreateRestitutionAttr().Set(RESTITUTION)
    if _HAS_PHYSX:
        try:
            pxm = PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
            pxm.CreateFrictionCombineModeAttr().Set('max')
            pxm.CreateRestitutionCombineModeAttr().Set('min')
        except Exception:
            mat_prim.CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set('max')
            mat_prim.CreateAttribute('physxMaterial:restitutionCombineMode', Sdf.ValueTypeNames.Token).Set('min')
    try:
        UsdShade.MaterialBindingAPI.Apply(collision_prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
    except Exception:
        rel = collision_prim.GetRelationship('physics:material:binding')
        if not rel:
            rel = collision_prim.CreateRelationship('physics:material:binding', custom=False)
        rel.SetTargets([mat_prim.GetPath()])

def _bind_visual(stage, prim, name: str, color: tuple, roughness: float=0.6) -> None:
    vis_mat = UsdShade.Material.Define(stage, f'/{name}/VisMaterial')
    shader = UsdShade.Shader.Define(stage, f'/{name}/VisMaterial/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    vis_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI(prim).Bind(vis_mat)

def _world_size(scale_yup: tuple[float, float, float]) -> list[float]:
    """Y-up authored extents -> world extents after the 90-degree X spawn rotation.

    +90 about X maps (x, y, z) -> (x, -z, y), so world_z is the authored Y. That is the same
    index ``sim_common.read_spawn_z_offset`` treats as the height, which is what keeps the two
    consistent.
    """
    (sx, sy, sz) = scale_yup
    return [sx, sz, sy]

def make_static_box_usd(output_dir: Path, name: str, color: tuple, scale_yup: tuple[float, float, float], *, collider: bool=True, extra_meta: dict | None=None, roughness: float=0.6) -> Path:
    """Author one static collider box. No RigidBodyAPI — see the module docstring."""
    output_dir.mkdir(parents=True, exist_ok=True)
    usd_path = output_dir / f'{name}.usd'
    if usd_path.exists():
        usd_path.unlink()
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f'/{name}')
    stage.SetDefaultPrim(root.GetPrim())
    cube = UsdGeom.Cube.Define(stage, f'/{name}/Mesh')
    cube.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*scale_yup))
    _bind_visual(stage, cube.GetPrim(), name, color, roughness)
    if collider:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _add_static_physics_material(stage, root.GetPrim().GetPath(), cube.GetPrim())
    stage.GetRootLayer().Save()
    dims = list(scale_yup)
    meta = {'bbox_size_m': dims, 'bbox_longest_edge_m': max(dims), 'world_size_m': _world_size(scale_yup), 'meters_per_unit': 1.0, 'static': True, 'up_axis': 'Y'}
    if extra_meta:
        meta.update(extra_meta)
    usd_path.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2))
    return usd_path
TABLE_BACK_MARGIN_M = 0.06

def table_extents() -> tuple[float, float, float, float]:
    pass
    return (-TABLE_BACK_MARGIN_M, WORKSPACE.x_max + TABLE_MARGIN_M, WORKSPACE.y_min - TABLE_MARGIN_M, WORKSPACE.y_max + TABLE_MARGIN_M)

def table_centre_xy() -> tuple[float, float]:
    """World XY centre of the slab. Consumers read this rather than recomputing it."""
    (x0, x1, y0, y1) = table_extents()
    return (round(0.5 * (x0 + x1), 6), round(0.5 * (y0 + y1), 6))

def table_scale() -> tuple[float, float, float]:
    """Y-up scale for the tabletop slab."""
    (x0, x1, y0, y1) = table_extents()
    return (x1 - x0, TABLE_THICKNESS_M, y1 - y0)

def _fmt_mm(h: float) -> str:
    return f'{int(round(h * 1000)):04d}'

def generate(output_dir: Path, *, dry_run: bool=False) -> dict[str, list[str]]:
    pass
    written: dict[str, list[str]] = {'table': []}
    tscale = table_scale()
    if not dry_run:
        make_static_box_usd(output_dir, 'escape_ood_table', TABLE_COLOR, tscale, extra_meta={'role': 'table', 'top_z_offset_m': TABLE_THICKNESS_M * 0.5, 'world_centre_xy': list(table_centre_xy()), 'world_extents_xy': list(table_extents())})
    written['table'].append(f'escape_ood_table  {[round(v, 3) for v in _world_size(tscale)]} m (world)')
    return written

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(OUTPUT_DIR), help='output directory for the USDs')
    ap.add_argument('--dry-run', action='store_true', help='list what would be written without authoring anything')
    args = ap.parse_args()
    out = Path(args.out)
    written = generate(out, dry_run=args.dry_run)
    tag = '[DRY]' if args.dry_run else '[OK] '
    print(f'\n{tag} output dir: {out}')
    for (role, items) in written.items():
        print(f'\n{tag} {role} ({len(items)}):')
        for line in items:
            print(f'       {line}')
    total = sum((len(v) for v in written.values()))
    print(f"\n{tag} {total} prop(s); physx schema: {('yes' if _HAS_PHYSX else 'no (raw attrs)')}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
