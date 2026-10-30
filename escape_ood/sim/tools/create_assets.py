import json
import sys
from pathlib import Path
try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    print('[ERROR] pxr not found. Activate the project venv and retry.', flush=True)
    sys.exit(1)
try:
    from pxr import PhysxSchema
    _HAS_PHYSX = True
except ImportError:
    _HAS_PHYSX = False
_REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = _REPO_ROOT / 'data' / 'output' / 'assets' / 'escape_ood_assets'
GLB_DIR = OUTPUT_DIR / 'glbs'
CUBES = [('escape_ood_cube_red', (1.0, 0.08, 0.08), (0.05, 0.05, 0.05))]
MASS_KG = 0.1
STATIC_FRICTION = 2.0
DYNAMIC_FRICTION = 2.0
RESTITUTION = 0.0

def _add_physics_material(stage: 'Usd.Stage', root_path: 'Sdf.Path', collision_prim: 'Usd.Prim'):
    """Create a UsdPhysics material and bind it to the collision prim."""
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
            if hasattr(pxm, 'CreateImprovePatchFrictionAttr'):
                pxm.CreateImprovePatchFrictionAttr().Set(True)
            else:
                mat_prim.CreateAttribute('physxMaterial:improvePatchFriction', Sdf.ValueTypeNames.Bool).Set(True)
        except Exception:
            mat_prim.CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set('max')
            mat_prim.CreateAttribute('physxMaterial:restitutionCombineMode', Sdf.ValueTypeNames.Token).Set('min')
            mat_prim.CreateAttribute('physxMaterial:improvePatchFriction', Sdf.ValueTypeNames.Bool).Set(True)
    try:
        UsdShade.MaterialBindingAPI.Apply(collision_prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
    except Exception:
        rel = collision_prim.GetRelationship('physics:material:binding')
        if not rel:
            rel = collision_prim.CreateRelationship('physics:material:binding', custom=False)
        rel.SetTargets([mat_prim.GetPath()])

def _add_ccd(root_prim: 'Usd.Prim'):
    """Apply CCD and solver iteration settings (PhysX-specific, best-effort)."""
    if not _HAS_PHYSX:
        root_prim.CreateAttribute('physxRigidBody:enableCCD', Sdf.ValueTypeNames.Bool).Set(True)
        root_prim.CreateAttribute('physxRigidBody:enableSpeculativeCCD', Sdf.ValueTypeNames.Bool).Set(True)
        root_prim.CreateAttribute('physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int).Set(14)
        root_prim.CreateAttribute('physxRigidBody:solverVelocityIterationCount', Sdf.ValueTypeNames.Int).Set(4)
        return
    try:
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
        if hasattr(rb, 'CreateEnableCCDAttr'):
            rb.CreateEnableCCDAttr(True)
        else:
            root_prim.CreateAttribute('physxRigidBody:enableCCD', Sdf.ValueTypeNames.Bool).Set(True)
        if hasattr(rb, 'CreateEnableSpeculativeCCDAttr'):
            rb.CreateEnableSpeculativeCCDAttr(True)
        else:
            root_prim.CreateAttribute('physxRigidBody:enableSpeculativeCCD', Sdf.ValueTypeNames.Bool).Set(True)
        if hasattr(rb, 'CreateSolverPositionIterationCountAttr'):
            rb.CreateSolverPositionIterationCountAttr(14)
        else:
            root_prim.CreateAttribute('physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int).Set(14)
        if hasattr(rb, 'CreateSolverVelocityIterationCountAttr'):
            rb.CreateSolverVelocityIterationCountAttr(4)
        else:
            root_prim.CreateAttribute('physxRigidBody:solverVelocityIterationCount', Sdf.ValueTypeNames.Int).Set(4)
    except Exception:
        root_prim.CreateAttribute('physxRigidBody:enableCCD', Sdf.ValueTypeNames.Bool).Set(True)
        root_prim.CreateAttribute('physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int).Set(14)

def make_block_usd(output_dir: Path, name: str, color: tuple, scale: tuple):
    usd_path = output_dir / f'{name}.usd'
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f'/{name}')
    stage.SetDefaultPrim(root.GetPrim())
    cube = UsdGeom.Cube.Define(stage, f'/{name}/Mesh')
    cube.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*scale))
    vis_mat = UsdShade.Material.Define(stage, f'/{name}/VisMaterial')
    shader = UsdShade.Shader.Define(stage, f'/{name}/VisMaterial/Shader')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    vis_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(vis_mat)
    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr().Set(MASS_KG)
    _add_ccd(root.GetPrim())
    _add_physics_material(stage, root.GetPrim().GetPath(), cube.GetPrim())
    stage.GetRootLayer().Save()
    dims = list(scale)
    meta = {'bbox_size_m': dims, 'bbox_longest_edge_m': max(dims), 'meters_per_unit': 1.0}
    meta_path = usd_path.with_suffix('.meta.json')
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'[OK] {usd_path.name}  bbox={[round(d * 100, 1) for d in dims]} cm', flush=True)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for (name, color, scale) in CUBES:
        make_block_usd(OUTPUT_DIR, name, color, scale)
if __name__ == '__main__':
    main()
