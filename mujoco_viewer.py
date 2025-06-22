"""
MuJoCo Environment Tester for SfM Reconstructions

Tests environments created from Structure from Motion data in MuJoCo physics simulator.
Uses separate meshes for each component to avoid convex hull collision issues.
"""

import numpy as np
import open3d as o3d
import mujoco
import time
from pathlib import Path
try:
    from deepsfm.mesh_creator import create_rl_environment
except ImportError:
    print("Error: Could not import mesh_creator module")
    print("Make sure the RLMeshCreator class is saved as 'mesh_creator.py'")


def sfm_to_mujoco(
    data_path='camera_view_ground_env.npz', 
    enable_env_collision=True,
    visualize_meshes=False,
    test_physics=True
):
    data = np.load(data_path)
    normalized_data = {
        'points': data['points'],
        'colors': data['colors'],
        'camera_positions': data.get('camera_positions', []),
        'camera_directions': data.get('camera_directions', [])
    }
    
    env_data = create_rl_environment(
        normalized_data, 
        add_walls=True, 
        simplify=True, 
        visualize=visualize_meshes
    )
    
    if not test_physics:
        print("\nSkipping physics test.")
        return
    
    tester = MuJoCoEnvironmentTester(env_data['meshes'])
    success = tester.test_environment(
        enable_env_collision=enable_env_collision,
        visualize=True
    )
    
    return success


if __name__ == "__main__":
    sfm_to_mujoco(enable_env_collision=True)