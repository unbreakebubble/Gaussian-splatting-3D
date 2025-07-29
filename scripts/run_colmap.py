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
    parser.add_argument('--use_metadata', action='store_true', help="Use image metadata from JSON files for better initialization")
    args = parser.parse_args()

    # Initialize using metadata if available
    if args.use_metadata:
        from prepare_metadata import extract_metadata, write_colmap_format, write_colmap_ini
        print("Extracting metadata from JSON files...")
        metadata = extract_metadata(args.images)
        if metadata:
            sparse_init_path = os.path.join(args.output, "sparse_init")
            write_colmap_format(metadata, sparse_init_path)
            write_colmap_ini(sparse_init_path, os.path.abspath(args.images))
            print(f"Initialized reconstruction using metadata from {len(metadata)} images")

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
        "--ImageReader.single_camera", "1",               # Skydio drone uses same camera
        "--ImageReader.camera_model", "SIMPLE_RADIAL",    # Radial distortion model for drone camera
        "--SiftExtraction.max_image_size", "3000",       # Limit max image size for better performance
        "--SiftExtraction.edge_threshold", "10",         # Increase edge threshold for drone imagery
        "--SiftExtraction.peak_threshold", "0.01",       # Lower peak threshold for better feature detection
        "--SiftExtraction.max_num_features", "8000"      # Increase max features for better coverage
    ]
    if args.use_gpu:
        feature_cmd += ["--SiftExtraction.use_gpu", "1"]
    else:
        feature_cmd += ["--SiftExtraction.use_gpu", "0"]
    run_cmd(feature_cmd)

    # 2. Image matching
    match_cmd = [
        colmap_path, "exhaustive_matcher",
        "--database_path", os.path.join(output_path, "database.db"),
        "--SiftMatching.guided_matching", "1",           # Enable guided matching for better accuracy
        "--SiftMatching.max_num_matches", "32000",       # Increase max matches for better coverage
    ]
    if args.use_gpu:
        match_cmd += ["--SiftMatching.use_gpu", "1"]
    run_cmd(match_cmd)

    # 3. Sparse mapping (reconstruction)
    sparse_path = os.path.join(output_path, "sparse")
    os.makedirs(sparse_path, exist_ok=True)
    
    # Base mapper command
    map_cmd = [
        colmap_path, "mapper",
        "--database_path", os.path.join(output_path, "database.db"),
        "--image_path", images_path,
        "--output_path", sparse_path
    ]
    
    # If we have metadata, use it for initialization
    if args.use_metadata and os.path.exists(os.path.join(output_path, "sparse_init")):
        map_cmd.extend([
            "--Mapper.init_min_tri_angle", "4",
            "--Mapper.multiple_models", "0",
            "--Mapper.extract_colors", "1",
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_extra_params", "1",
            "--Mapper.init_existing_model", "1",
            "--input_path", os.path.join(output_path, "sparse_init")
        ])
    
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
            "-n", "4000",              # Increase number of iterations
        ]
        run_cmd(opensplat_cmd)
        print("OpenSplat completed. Results saved to splat.ply.")
    else:
        print(f"OpenSplat not found at {opensplat_exe}")
        print("Please make sure OpenSplat is built and the path is correct.")

if __name__ == "__main__":
    main()