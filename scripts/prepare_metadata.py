import os
import json
import glob
import math

def extract_metadata(images_path):
    """Extract metadata from JSON files and write COLMAP-compatible format."""
    image_metadata = []
    sample_image = None
    
    # First pass to get camera parameters from first image
    for img_path in glob.glob(os.path.join(images_path, "*.JP*G")):
        json_path = img_path.rsplit('.', 1)[0] + ".JSON"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                sample_image = json.load(f)
            break
    
    # Get camera parameters from sample image
    if sample_image:
        width = int(sample_image.get("PixelXDimension", 4056))
        height = int(sample_image.get("PixelYDimension", 3040))
        
        # Handle focal length that might be in fraction format (e.g. "3700/1000")
        focal_raw = sample_image.get("FocalLength", "3700/1000")
        if isinstance(focal_raw, str) and '/' in focal_raw:
            num, denom = map(float, focal_raw.split('/'))
            focal = num / denom
        else:
            focal = float(focal_raw)
            
        optical_center_x = float(sample_image.get("CalibratedOpticalCenter", {}).get("X", width/2))
        optical_center_y = float(sample_image.get("CalibratedOpticalCenter", {}).get("Y", height/2))
        
        # Handle distortion parameters that might be in comma-separated format
        dewarp_data = sample_image.get("DewarpData", "0.123224,0,0")
        radial_distortion = float(dewarp_data.split(",")[0])
    else:
        # Default values if no metadata found
        width, height = 4056, 3040
        focal = 3700.0
        optical_center_x, optical_center_y = 2022.872267, 1512.190903
        radial_distortion = 0.123224
    
    # Store camera parameters
    global cameras_text
    cameras_text = [f"1 SIMPLE_RADIAL {width} {height} {focal} {optical_center_x} {optical_center_y} {radial_distortion}"]
    
    for img_path in glob.glob(os.path.join(images_path, "*.JP*G")):
        json_path = img_path.rsplit('.', 1)[0] + ".JSON"
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                
            # Extract relevant information
            lat = float(metadata.get("Latitude", 0))
            lon = float(metadata.get("Longitude", 0))
            alt = float(metadata.get("AbsoluteAltitude", 0))
            
            # Get camera orientation
            orientation = metadata.get("CameraOrientationNED", {})
            yaw = float(orientation.get("Yaw", 0))
            pitch = float(orientation.get("Pitch", 0))
            roll = float(orientation.get("Roll", 0))
            
            # Camera calibration info if available
            focal_raw = sample_image.get("FocalLength", "3700/1000")
            if isinstance(focal_raw, str) and '/' in focal_raw:
                num, denom = map(float, focal_raw.split('/'))
                focal = num / denom
            else:
                focal = float(focal_raw)
            
            image_metadata.append({
                "image_name": os.path.basename(img_path),
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "focal_length": focal
            })
    
    return image_metadata

def write_colmap_format(metadata, output_dir):
    """Write metadata in COLMAP format for initialization."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Write cameras.txt
    with open(os.path.join(output_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(cameras_text[0] + "\n")
    
    # Write images.txt
    with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for idx, data in enumerate(metadata):
            # Convert NED orientation to quaternion (simplified)
            # In real implementation, you'd want a proper Euler to quaternion conversion
            # yaw_rad = math.radians(data['yaw'])
            # pitch_rad = math.radians(data['pitch'])
            # roll_rad = math.radians(data['roll'])
            
            # Simplified position conversion from GPS
            # In real implementation, you'd want proper GPS to local coordinate conversion
            tx = data['longitude'] * 111000  # Approximate meters
            ty = data['latitude'] * 111000   # Approximate meters
            tz = data['altitude']
            
            f.write(f"{idx+1} 1 0 0 0 {tx} {ty} {tz} 1 {data['image_name']}\n")
            f.write("\n")  # Empty line for POINTS2D (will be filled by COLMAP)

def write_colmap_ini(output_dir, image_path):
    """Write a COLMAP project configuration file."""
    config = f"""
[General]
database_path=database.db
image_path={image_path}
[Mapper]
init_min_tri_angle=4
multiple_models=0
extract_colors=1
ba_refine_focal_length=1
ba_refine_extra_params=1
min_focal_length_ratio=0.1
max_focal_length_ratio=10
max_extra_param=1
"""
    with open(os.path.join(output_dir, 'colmap.ini'), 'w') as f:
        f.write(config)

if __name__ == "__main__":
    import math
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare metadata for COLMAP reconstruction")
    parser.add_argument('--images', required=True, help="Path to input images folder")
    parser.add_argument('--output', default="sparse", help="Output directory for COLMAP files")
    args = parser.parse_args()
    
    metadata = extract_metadata(args.images)
    if metadata:
        write_colmap_format(metadata, args.output)
        write_colmap_ini(args.output, os.path.abspath(args.images))
        print(f"Successfully processed metadata for {len(metadata)} images")
        print(f"COLMAP format files written to: {args.output}")
    else:
        print("No metadata found!")
