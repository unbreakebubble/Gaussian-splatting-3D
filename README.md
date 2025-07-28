# Gaussian Splatting 3D

This project provides an end-to-end pipeline for reconstructing a 3D scene from drone images using Gaussian Splatting. The pipeline integrates Structure-from-Motion (SfM) using COLMAP and visualizes the results with an interactive Three.js viewer.

## Project Structure

```
gaussian-splatting-3d
├── scripts
│   └── run_colmap.py          # Script for automating COLMAP feature extraction and mapping
├── src
│   └── __init__.py            # Initialization file for the src package
├── data
│   └── images                 # Directory for input drone images
├── output
│   ├── images                 # Directory for processed images
│   ├── sparse                 # Directory for sparse reconstruction results
│   ├── database.db            # COLMAP database with features and matches
│   ├── splat.ply              # Resulting Gaussian splat point cloud in PLY format
│   └── cameras.json           # Camera calibration and pose information
├── viewer.html                # HTML file for visualizing the Gaussian splat point cloud
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```

## Features

- **Automatic COLMAP Reconstruction**: Estimates camera poses and creates a sparse 3D point cloud from input images.
- **Gaussian Splatting Conversion**: Converts the sparse reconstruction into an optimized set of 3D Gaussian splats.
- **Interactive Visualization**: Provides a web-based viewer to explore the resulting 3D model.

## Prerequisites

Before running the pipeline, ensure you have the following installed:

- **COLMAP**: For feature extraction and mapping. Install via package manager or download from the COLMAP website.
- **OpenSplat** (optional): For Gaussian Splatting. Recommended to use the Dockerized version for ease of setup.
- **Nerfstudio** (optional): An alternative to OpenSplat if needed.

## Setup Instructions

1. **Clone the Repository**:
   ```
   git clone https://github.com/unbreakebubble/Gaussian-splatting-3D.git
   cd Gaussian-splatting-3D
   ```

2. **Prepare Input Images**: Organize your drone images into the `data/images` directory.

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Running the Pipeline

1. **Feature Extraction & Sparse Reconstruction**:
   Run the COLMAP script:
   ```
   python scripts/run_colmap.py --images path/to/your/images --output output
   ```

2. **Gaussian Splatting Optimization**:
   Use OpenSplat to convert the sparse points into Gaussians:
   ```
   docker run -it -v $(pwd)/output:/data opensplat bash -c "cd /code/build && ./opensplat /data -n 2000"
   ```

3. **Visualization**:
   Start a local web server:
   ```
   python3 -m http.server 8000
   ```
   Open your browser at `http://localhost:8000/viewer.html` to view the 3D model.

## Troubleshooting Tips

- Ensure images have sufficient overlap for COLMAP to find features.
- Adjust the number of iterations in OpenSplat for better quality.
- Use the COLMAP GUI to inspect the database if reconstruction fails.

## Conclusion

This pipeline allows you to create a 3D model from drone images efficiently. Feel free to modify the code and explore the capabilities of Gaussian Splatting for your projects.