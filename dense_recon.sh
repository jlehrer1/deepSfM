#!/bin/bash

# Ensure that the user provides an input path to the sparse reconstruction
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_sparse_reconstruction>"
  exit 1
fi

# Set the path to the sparse reconstruction folder
SPARSE_PATH="$1"

# Define the path to COLMAP executable (make sure this points to your COLMAP installation)
COLMAP_PATH="colmap"  # Change this to your COLMAP binary path

# Set output directories for dense reconstruction
OUTPUT_PATH="${SPARSE_PATH}/dense"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Step 1: Run COLMAP dense stereo
"$COLMAP_PATH" patch_match_stereo \
  --workspace_path "$SPARSE_PATH" \
  --PatchMatchStereo.max_image_size 4000 \
  --PatchMatchStereo.max_iterations 6 \
  --PatchMatchStereo.geom_consistency true

# Step 2: Run COLMAP dense fusion
"$COLMAP_PATH" stereo_fusion \
  --workspace_path "$SPARSE_PATH" \
  --output_path "$OUTPUT_PATH" \
  --StereoFusion.min_num_pixels 5 \
  --StereoFusion.max_reproj_error 2

# Print success message
echo "Dense reconstruction completed. Results saved in: $OUTPUT_PATH"
