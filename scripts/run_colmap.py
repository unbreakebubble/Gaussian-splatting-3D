#!/usr/bin/env python3
"""
run_colmap.py: Automate COLMAP feature extraction, matching, and mapping for a given image set.
Usage:
    python run_colmap.py --images <image_folder> [--output <output_folder>]
"""

import os
import shutil
import subprocess
import argparse

def run_cmd(cmd, cwd=None):
    """Helper to run a system command and check for errors."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with code {result.returncode}: {' '.join(cmd)}")

def main():
    parser = argparse.ArgumentParser(description="Run COLMAP on a set of images to produce a sparse model.")
    parser.add_argument('--images', required=True, help="Path to input images folder")
    parser.add_argument('--output', default="output", help="Output directory (will be created if not exists)")
    parser.add_argument('--use_gpu', action='store_true', help="Use GPU for feature extraction and matching (if COLMAP is compiled with CUDA).")
    args = parser.parse_args()

    images_path = os.path.abspath(args.images)
    output_path = os.path.abspath(args.output)
    os.makedirs(output_path, exist_ok=True)

    # Find COLMAP executable
    colmap_path = r"C:\Users\upadh\OneDrive\Desktop\Repos\temp\vcpkg\installed\x64-windows\tools\colmap\colmap.exe"
    if not os.path.exists(colmap_path):
        raise SystemExit("Could not find COLMAP executable. Please make sure COLMAP is installed correctly.")

    # 1. Feature extraction
    feature_cmd = [
        colmap_path, "feature_extractor",
        "--database_path", os.path.join(output_path, "database.db"),
        "--image_path", images_path,
        "--ImageReader.single_camera", "1",               # assume one camera (e.g., same drone camera for all images)
        "--ImageReader.camera_model", "SIMPLE_RADIAL"     # use simple pinhole model (good for most cameras)
    ]
    if args.use_gpu:
        feature_cmd += ["--SiftExtraction.use_gpu", "1"]
    else:
        feature_cmd += ["--SiftExtraction.use_gpu", "0"]
    run_cmd(feature_cmd)

    # 2. Image matching
    match_cmd = [
        colmap_path, "exhaustive_matcher",
        "--database_path", os.path.join(output_path, "database.db")
    ]
    if args.use_gpu:
        match_cmd += ["--SiftMatching.use_gpu", "1"]
    run_cmd(match_cmd)

    # 3. Sparse mapping (reconstruction)
    sparse_path = os.path.join(output_path, "sparse")
    os.makedirs(sparse_path, exist_ok=True)
    map_cmd = [
        colmap_path, "mapper",
        "--database_path", os.path.join(output_path, "database.db"),
        "--image_path", images_path,
        "--output_path", sparse_path
    ]
    run_cmd(map_cmd)

    # 4. Copy images into output folder (for OpenSplat compatibility)
    dest_images = os.path.join(output_path, "images")
    if not os.path.exists(dest_images):
        shutil.copytree(images_path, dest_images)
        print(f"Copied images to {dest_images}")
    else:
        print(f"Images folder already exists in output (skipped copying).")

    print("\nCOLMAP reconstruction completed. Sparse model is in:", sparse_path)

    # 5. Run OpenSplat to generate splat model
    opensplat_exe = r"C:\Users\upadh\OneDrive\Desktop\Repos\OpenSplat\build\Release\opensplat.exe"
    if os.path.exists(opensplat_exe):
        print("OpenSplat found. Running Gaussian splatting optimization...")
        # Enhanced OpenSplat parameters for better quality
        opensplat_cmd = [
            opensplat_exe, 
            output_path,
            "--output", os.path.join(output_path, "splat.ply"),
            "-n", "10000",
            "-s", "1000",       # Increase number of iterations
        ]
        run_cmd(opensplat_cmd)
        print("OpenSplat completed. Results saved to splat.ply.")
    else:
        print("OpenSplat not found in PATH. Please run the OpenSplat tool manually to generate the splat model.")
        print(f"Example (from repo root or OpenSplat directory): ./opensplat {output_path} -n 2000")

if __name__ == "__main__":
    main()