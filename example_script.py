import pycolmap
from deepsfm.deepsfm import reconstruct_images
from deepsfm.visualize3d import plot_reconstruction, init_figure
from pathlib import Path
from torchvision.transforms import Resize, Compose, PILToTensor
import plotly.io as pio
import open3d as o3d 
import os 
import numpy as np

import pycolmap
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from deepsfm.mesh_creator import CameraViewGroundDetector, create_and_preview_rl_environment
import sys
import os
import numpy as np
import open3d as o3d
import habitat_sim
import magnum as mn
from habitat_sim.utils import viz_utils as vut

import os
import subprocess

# Make sure to get the libomp path using brew
libomp_prefix = subprocess.check_output(['brew', '--prefix', 'libomp']).decode().strip()

from deepsfm import reconstruct_images
from deepsfm.visualize3d import plot_reconstruction, init_figure

img_path = "images"
recon = reconstruct_images(img_path, single_camera=False)
fig = init_figure()
plot_reconstruction(fig, recon, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
fig.show()

# # Set environment variables
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# os.environ['DYLD_LIBRARY_PATH'] = f"{libomp_prefix}/lib:" + os.environ.get('DYLD_LIBRARY_PATH', '')

# print("Environment variables set:")
# print(f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
# print(f"KMP_DUPLICATE_LIB_OK={os.environ['KMP_DUPLICATE_LIB_OK']}")
# print(f"DYLD_LIBRARY_PATH={os.environ['DYLD_LIBRARY_PATH']}")
        
# # Main execution
# recon_path = "reconstruction/sparse/0"

# detector = CameraViewGroundDetector(recon_path)
# results = detector.detect_ground_and_normalize()

# np.savez('camera_view_ground_env.npz',
#     points=results['normalized_data']['points'],
#     colors=results['normalized_data']['colors'],
#     camera_positions=results['normalized_data']['camera_positions'],
#     camera_directions=results['normalized_data']['camera_directions'])

# # Load and create RL environment
# data = np.load('camera_view_ground_env.npz')
# normalized_data = {
#     'points': data['points'],
#     'colors': data['colors'],
#     'camera_positions': data['camera_positions'],
#     'camera_directions': data['camera_directions']
# }
# rl_env, stats = create_and_preview_rl_environment(normalized_data, add_walls=True, visualize=True)
