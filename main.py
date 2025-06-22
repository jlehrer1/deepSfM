"""
MuJoCo Humanoid Walking Training for SfM Reconstructions

Trains a humanoid agent to walk in environments created from Structure from Motion data.
Uses PPO (Proximal Policy Optimization) with actor-critic networks.
"""

import numpy as np
from deepsfm.mesh_creator import create_rl_environment
from deepsfm.mujoco_envs import HumanoidEnvironmentTester, HumanoidWalkingEnv, MuJoCoEnvironmentTester
from deepsfm.actor_critic import PPOTrainer
import os
from deepsfm.mesh_creator import CameraViewGroundDetector
from deepsfm.deepsfm import reconstruct_images

TRAIN_WALKING_MODEL = False 

data_path = "camera_view_ground_env.npz"
img_path = "images"

recon = reconstruct_images(img_path, single_camera=True)
recon_path = "/Users/julianlehrer/Documents/Projects/deepSfM/reconstruction/sparse/0"

detector = CameraViewGroundDetector(recon_path)
results = detector.detect_ground_and_normalize()

np.savez(
    "camera_view_ground_env.npz",
    points=results["normalized_data"]["points"],
    colors=results["normalized_data"]["colors"],
    camera_positions=results["normalized_data"]["camera_positions"],
    camera_directions=results["normalized_data"]["camera_directions"],
)

visualize_meshes = False
training_timesteps = 1000000
data = np.load(data_path)
normalized_data = {
    "points": data["points"],
    "colors": data["colors"],
    "camera_positions": data.get("camera_positions", []),
    "camera_directions": data.get("camera_directions", []),
}

env_data = create_rl_environment(normalized_data, add_walls=True, simplify=True, visualize=visualize_meshes)

tester = MuJoCoEnvironmentTester(env_data['meshes'])
success = tester.test_environment(
    enable_env_collision=True,
    visualize=True
)

if TRAIN_WALKING_MODEL:
    tester = HumanoidEnvironmentTester(env_data["meshes"])

    if not tester.save_meshes():
        print("Failed to save meshes!")

    xml = tester.create_humanoid_mjcf_xml()
    xml_path = tester.mesh_dir / "humanoid_environment.xml"
    with open(xml_path, "w") as f:
        f.write(xml)
    print(f"Saved MuJoCo XML: {xml_path}")

    env = HumanoidWalkingEnv(xml)
    trainer = PPOTrainer(env)
    trainer.train(total_timesteps=training_timesteps)
