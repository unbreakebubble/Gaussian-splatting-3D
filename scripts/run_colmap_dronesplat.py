#!/usr/bin/env python3
"""
fixed_colmap_dronesplat.py — zero‑arg pipeline using DroneSplat (rev‑7)
======================================================================
This **fully replaces OpenSplat** with the official CVPR‑25 **DroneSplat**
implementation (https://github.com/BITyia/DroneSplat).  It still relies on
COLMAP for initial camera poses but hands everything else to DroneSplat.

-----------------------------------------------------------------------
🔧 One‑time installation (Windows‑CUDA example)
-----------------------------------------------------------------------
```powershell
# 1. Install DroneSplat and deps
cd C:\Users\upadh\Repos
git clone --recursive https://github.com/BITyia/DroneSplat.git
cd DroneSplat
conda create -n dronesplat python=3.11 -y
conda activate dronesplat
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install submodules/simple-knn submodules/diff-gaussian-rasterization
# optional (faster masks):
cd submodules\sam2 && pip install -e . && cd .. ..

# 2. Pull SAM2 + DUSt3R checkpoints (see README) into DroneSplat\checkpoints\

# 3. Build COLMAP (already done) and note path.
```

-----------------------------------------------------------------------
▶️  To process a new scene
-----------------------------------------------------------------------
```powershell
python fixed_colmap_dronesplat.py  data\my_roof_images   work\roof_demo
start work\roof_demo\viewer\index.html  # view in browser
```

No CLI flags.  Edit the constants at the top if you move executables.
"""
from __future__ import annotations
import json, shutil, subprocess, sys, os
from pathlib import Path
from typing import Tuple

try:
    from PIL import Image
except ImportError:
    sys.exit("Run `pip install pillow` inside your env first.")

# ---------------------------------------------------------------------------
# Hard‑wired executables
COLMAP_EXE     = Path(r"C:\Users\upadh\OneDrive\Desktop\Repos\temp\vcpkg\installed\x64-windows\tools\colmap\colmap.exe")
DRONESPLAT_ROOT = Path(r"C:\Users\upadh\Repos\DroneSplat")            # repo root
PY             = Path(sys.executable)                                     # current python (should be dronesplat env)

MAX_IMAGE_SIZE   = 3200
SPATIAL_NEIGHBORS = 40
SPATIAL_RADIUS_M  = 12.0
DRONESPLAT_ITERS  = 7000    # default from README (can raise later)
USE_GPU           = True
# ---------------------------------------------------------------------------

def pick_json(img_dir: Path) -> dict:
    for j in img_dir.glob('*.json'):
        return json.loads(j.read_text())
    raise FileNotFoundError('No Skydio JSON metadata found')

def pick_sample(img_dir: Path) -> Path:
    for ext in ('*.JPG','*.jpg','*.png','*.jpeg'):
        files = list(img_dir.glob(ext))
        if files:
            return files[0]
    raise FileNotFoundError('No images found')

def intrinsics(meta, img_path: Path) -> Tuple[str,float]:
    w,h = Image.open(img_path).size
    fx  = float(meta['CalibratedFocalLength']['X'])
    cx  = float(meta['CalibratedOpticalCenter']['X'])
    cy  = float(meta['CalibratedOpticalCenter']['Y'])
    k1  = 0.0
    return f"{fx},{cx},{cy},{k1}", fx/max(w,h)

def run(cmd, cwd=None):
    print('»', ' '.join(map(str,cmd)))
    subprocess.check_call(list(map(str,cmd)), cwd=cwd)

# ---------------------------------------------------------------------------

def main():
    if len(sys.argv)!=3:
        sys.exit('Usage: python fixed_colmap_dronesplat.py <images_dir> <output_dir>')

    images = Path(sys.argv[1]).resolve()
    out    = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not COLMAP_EXE.exists():
        sys.exit('COLMAP executable path invalid')
    if not (DRONESPLAT_ROOT/'train.py').exists():
        sys.exit('DroneSplat repo not found; fix DRONESPLAT_ROOT constant')

    meta = pick_json(images)
    sample = pick_sample(images)
    cam_params, foc_factor = intrinsics(meta,sample)

    # 1) COLMAP ---------------------------------------------------------
    db = out/'colmap.db'
    run([
        COLMAP_EXE,'feature_extractor',
        '--database_path',db,
        '--image_path',images,
        '--ImageReader.camera_model','SIMPLE_RADIAL',
        '--ImageReader.single_camera','1',
        '--ImageReader.camera_params',cam_params,
        '--SiftExtraction.domain_size_pooling','1',
        '--SiftExtraction.estimate_affine_shape','1',
        '--SiftExtraction.max_image_size',str(MAX_IMAGE_SIZE),
        '--SiftExtraction.use_gpu','1' if USE_GPU else '0',
        '--ImageReader.default_focal_length_factor',f'{foc_factor:.6f}',
    ])

    run([
        COLMAP_EXE,'spatial_matcher',
        '--database_path',db,
        '--SpatialMatching.max_num_neighbors',str(SPATIAL_NEIGHBORS),
        '--SpatialMatching.max_distance',str(SPATIAL_RADIUS_M),
        '--SiftMatching.use_gpu','1' if USE_GPU else '0'
    ])

    sparse = out/'sparse'; sparse.mkdir(exist_ok=True)
    run([
        COLMAP_EXE,'mapper',
        '--database_path',db,
        '--image_path',images,
        '--output_path',sparse,
        '--Mapper.ba_refine_principal_point','0',
        '--Mapper.tri_ignore_two_view_tracks','1'
    ])

    # 2) Prepare DroneSplat scene dir ----------------------------------
    scene_dir = out/'dronesplat_scene'
    img_dst   = scene_dir/'images'
    if not img_dst.exists():
        img_dst.mkdir(parents=True, exist_ok=True)
        for p in images.glob('*.[jJ][pP][gG]'):
            shutil.copy2(p, img_dst/p.name)
        # copy COLMAP poses
    (scene_dir/'colmap').exists() or shutil.copytree(sparse, scene_dir/'colmap')

    # 3) (Optional) automatic masks via SAM2
    seg_script = DRONESPLAT_ROOT/'seg_all_instances.py'
    if seg_script.exists():
        try:
            run([PY, seg_script, '--image_dir', img_dst])
        except subprocess.CalledProcessError:
            print('⚠ seg_all_instances.py failed – continuing without masks')

    # 4) DroneSplat training -------------------------------------------
    model_out = out/'model'
    run([PY, DRONESPLAT_ROOT/'train.py',
         '-s', scene_dir,
         '-m', model_out,
         '--scene','myscene',
         '--iter', str(DRONESPLAT_ITERS),
         '--use_masks'])

    # 5) Render & export PLY -------------------------------------------
    ply_out = out/'dronesplat.ply'
    run([PY, DRONESPLAT_ROOT/'render.py',
         '-s', scene_dir,
         '-m', model_out,
         '--iter', str(DRONESPLAT_ITERS),
         '--export_ply', ply_out])

    # 6) Tiny web viewer ------------------------------------------------
    viewer = out/'viewer'; viewer.mkdir(exist_ok=True)
    viewer_html = viewer/'index.html'
    viewer_html.write_text(f"""<!doctype html><meta charset=utf-8><title>DroneSplat Viewer</title>
<script type=importmap>{{"imports":{{"three":"https://cdnjs.cloudflare.com/ajax/libs/three.js/0.174.0/three.module.js","@sparkjsdev/spark":"https://sparkjs.dev/releases/spark/0.1.6/spark.module.js"}}}}</script>
<style>html,body{{margin:0;height:100%;overflow:hidden}}</style>
<body><script type=module>
import * as THREE from 'three';
import {{OrbitControls}} from 'https://unpkg.com/three@0.174.0/examples/jsm/controls/OrbitControls.js';
import {{SplatMesh}} from '@sparkjsdev/spark';
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 1000);
camera.position.set(0,0,10);
const renderer = new THREE.WebGLRenderer(); document.body.appendChild(renderer.domElement);
new OrbitControls(camera, renderer.domElement);
scene.add(new SplatMesh({{url:'../dronesplat.ply'}}));
addEventListener('resize',()=>{{renderer.setSize(innerWidth,innerHeight); camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix();}});
window.dispatchEvent(new Event('resize'));
renderer.setAnimationLoop(()=>renderer.render(scene,camera));
</script>""")

    print('✔ Pipeline complete — view', viewer_html)

if __name__ == '__main__':
    main()
