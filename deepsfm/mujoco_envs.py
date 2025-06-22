import numpy as np
import open3d as o3d
import mujoco
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path

class MuJoCoEnvironmentTester:
    """Test SfM environments in MuJoCo physics simulator"""
    
    def __init__(self, meshes, mesh_dir="."):
        """
        Args:
            meshes: Dictionary of mesh objects from RLMeshCreator
            mesh_dir: Directory to save mesh files
        """
        self.meshes = meshes
        self.mesh_dir = Path(mesh_dir)
        self.mesh_files = {}
        self.bounds = self._calculate_bounds()
        
    def _calculate_bounds(self):
        """Calculate overall bounds of all meshes"""
        all_vertices = []
        for mesh in self.meshes.values():
            vertices = np.asarray(mesh.vertices)
            if len(vertices) > 0:
                all_vertices.append(vertices)
        
        if all_vertices:
            all_vertices = np.vstack(all_vertices)
            return {
                'min': all_vertices.min(axis=0),
                'max': all_vertices.max(axis=0)
            }
        return {'min': np.zeros(3), 'max': np.ones(3)}
    
    def save_meshes(self):
        """Save all meshes as OBJ files"""
        print("\nSaving meshes...")
        for name, mesh in self.meshes.items():
            filename = self.mesh_dir / f"{name}.obj"
            success = o3d.io.write_triangle_mesh(str(filename), mesh, write_ascii=True)
            if success:
                self.mesh_files[name] = filename.name
                print(f"  {name}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            else:
                print(f"  Failed to save {name}")
        return len(self.mesh_files) == len(self.meshes)
    
    def create_mjcf_xml(self, enable_env_collision=True, agent_radius=0.15):
        """
        Create MuJoCo XML string for the environment.
        
        Args:
            enable_env_collision: Whether environment mesh has collision
            agent_radius: Radius of test ball agent
        """
        # Calculate positions
        ground_z = self.bounds['min'][2]
        agent_start_z = self.bounds['max'][2] + 1.0
        
        # Start XML
        xml = f"""<mujoco model="sfm_environment">
    <compiler angle="degree" meshdir="{self.mesh_dir}" />
    <option timestep="0.01" gravity="0 0 -9.81"/>
    
    <asset>
"""
        
        # Add mesh assets
        for name, filename in self.mesh_files.items():
            xml += f'        <mesh name="{name}" file="{filename}"/>\n'
        
        xml += """    </asset>
    
    <worldbody>
"""
        
        # Add ground mesh
        if 'ground' in self.mesh_files:
            xml += """        <!-- Ground plane -->
        <geom name="ground" type="mesh" mesh="ground" pos="0 0 0" 
              rgba="0.4 0.3 0.2 1" 
              contype="1" conaffinity="1" 
              friction="1 0.5 0.5"/>
"""
        
        for i in range(4):
            if f'wall_{i}' in self.mesh_files:
                xml += f"""        
        <!-- Wall {i} -->
        <geom name="wall_{i}" type="mesh" mesh="wall_{i}" pos="0 0 0" 
              rgba="0.8 0.8 0.9 1" 
              contype="1" conaffinity="1" 
              friction="1 0.5 0.5"/>
"""
        
        if 'environment' in self.mesh_files:
            if enable_env_collision:
                xml += """        
        <!-- Environment mesh (building) WITH COLLISION -->
        <geom name="environment" type="mesh" mesh="environment" pos="0 0 0" 
              rgba="0.7 0.7 0.7 1" 
              contype="1" conaffinity="1"
              friction="1 0.5 0.5"
              condim="1"/>
"""
            else:
                xml += """        
        <!-- Environment mesh (building) VISUAL ONLY -->
        <geom name="environment" type="mesh" mesh="environment" pos="0 0 0" 
              rgba="0.7 0.7 0.7 1" 
              contype="0" conaffinity="0"/>
"""
        
        # Add test agent
        xml += f"""
        <!-- Test agent (ball) -->
        <body name="agent" pos="0 0 {agent_start_z}">
            <freejoint/>
            <geom name="agent_geom" type="sphere" size="{agent_radius}" 
                  rgba="1 0.2 0.2 1"
                  contype="2" conaffinity="1"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
        </body>
        
        <!-- Visual markers -->
        <geom name="ground_marker" type="cylinder" size="0.1 0.02" 
              pos="0 0 {ground_z + 0.02}" 
              rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    </worldbody>
    
    <contact>
        <pair geom1="agent_geom" geom2="ground"/>
"""
        
        # Add wall contacts
        for i in range(4):
            if f'wall_{i}' in self.mesh_files:
                xml += f'        <pair geom1="agent_geom" geom2="wall_{i}"/>\n'
        
        # Add environment contact if collision enabled
        if 'environment' in self.mesh_files and enable_env_collision:
            xml += '        <pair geom1="agent_geom" geom2="environment"/>\n'
        
        xml += """    </contact>
</mujoco>
"""
        
        return xml
    
    def run_drop_test(self, model, data, duration=50.0, log_interval=1.0):
        """Run physics simulation and log ball position"""
        print("\nRunning drop test...")
        
        steps_per_second = int(1.0 / model.opt.timestep)
        total_steps = int(duration * steps_per_second)
        log_steps = int(log_interval * steps_per_second)
        
        agent_start_z = data.qpos[2]
        ground_z = self.bounds['min'][2]
        expected_z = ground_z + 0.15  # Assuming ball radius of 0.15
        
        positions = []
        
        for step in range(total_steps):
            mujoco.mj_step(model, data)
            
            if step % log_steps == 0 or step == total_steps - 1:
                t = step / steps_per_second
                z = data.qpos[2]
                vz = data.qvel[2]
                positions.append((t, z, vz))
                print(f"  t={t:.1f}s: Z={z:.3f}, Vz={vz:.3f}")
        
        final_z = positions[-1][1]
        success = abs(final_z - expected_z) < 0.05
        return success, positions
    
    def test_environment(self, enable_env_collision=True, visualize=True):
        """
        Test the environment in MuJoCo.
        
        Args:
            enable_env_collision: Whether environment mesh has collision
            visualize: Whether to launch interactive viewer
        """
        # Save meshes
        if not self.save_meshes():
            print("Failed to save meshes!")
            return False
        
        # Create and save XML
        xml = self.create_mjcf_xml(enable_env_collision=enable_env_collision)
        xml_path = self.mesh_dir / "environment.xml"
        with open(xml_path, 'w') as f:
            f.write(xml)
        print(f"\nSaved MuJoCo XML: {xml_path}")
        
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        if visualize:
            try:
                # Try to get viewer function
                if hasattr(mujoco, 'viewer'):
                    viewer = mujoco.viewer.launch_passive(model, data)
                    
                    step = 0
                    agent_start_pos = data.qpos[:3].copy()
                    
                    while viewer.is_running():
                        # Reset every 5 seconds
                        if step % 500 == 0 and step > 0:
                            data.qpos[:3] = agent_start_pos
                            data.qpos[3:7] = [1, 0, 0, 0]  # Reset orientation
                            data.qvel[:] = 0
                            print(f"Reset ball to Z={agent_start_pos[2]:.3f}")
                        
                        mujoco.mj_step(model, data)
                        viewer.sync()
                        step += 1
                        time.sleep(0.01)
                else:
                    print("Note: Interactive viewer not available in this MuJoCo version")
                    
            except Exception as e:
                print(f"Viewer error: {e}")
        
        return True


class HumanoidEnvironmentTester:
    def __init__(self, meshes, mesh_dir="."):
        self.meshes = meshes
        self.mesh_dir = Path(mesh_dir)
        self.mesh_files = {}
        self.bounds = self._calculate_bounds()
        
    def _calculate_bounds(self):
        all_vertices = []
        for mesh in self.meshes.values():
            vertices = np.asarray(mesh.vertices)
            if len(vertices) > 0:
                all_vertices.append(vertices)
        
        if all_vertices:
            all_vertices = np.vstack(all_vertices)
            return {
                'min': all_vertices.min(axis=0),
                'max': all_vertices.max(axis=0)
            }
        return {'min': np.zeros(3), 'max': np.ones(3)}
    
    def save_meshes(self):
        """Save all meshes as OBJ files"""
        print("\nSaving meshes...")
        for name, mesh in self.meshes.items():
            filename = self.mesh_dir / f"{name}.obj"
            success = o3d.io.write_triangle_mesh(str(filename), mesh, write_ascii=True)
            if success:
                self.mesh_files[name] = filename.name
                print(f"  {name}: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            else:
                print(f"  Failed to save {name}")
        return len(self.mesh_files) == len(self.meshes)
    
    def create_humanoid_mjcf_xml(self):
        """Create MuJoCo XML with humanoid agent"""
        
        ground_z = self.bounds['min'][2]
        agent_start_z = ground_z + 1.5  # Start humanoid higher
        
        xml = f"""<mujoco model="humanoid_sfm_environment">
    <compiler angle="degree" meshdir="{self.mesh_dir}" />
    <option timestep="0.002" gravity="0 0 -9.81" iterations="50" solver="PGS"/>
    
    <default>
        <joint damping="1" stiffness="0" armature="0" limited="true" solreflimit="0.02" solimplimit="0.9"/>
        <geom condim="3" contype="1" conaffinity="1" friction="1 0.5 0.5" rgba="0.8 0.6 0.4 1"/>
        <motor ctrlrange="-1 1" ctrllimited="true"/>
    </default>
    
    <asset>
"""
        
        # Add mesh assets
        for name, filename in self.mesh_files.items():
            xml += f'        <mesh name="{name}" file="{filename}"/>\n'
        
        xml += """    </asset>
    
    <worldbody>
"""
        
        # Add ground mesh
        if 'ground' in self.mesh_files:
            xml += """        <!-- Ground plane -->
        <geom name="ground" type="mesh" mesh="ground" pos="0 0 0" 
              rgba="0.4 0.3 0.2 1" 
              contype="1" conaffinity="1" 
              friction="1 0.5 0.5"/>
"""
        
        # Add wall meshes
        for i in range(4):
            if f'wall_{i}' in self.mesh_files:
                xml += f"""        
        <!-- Wall {i} -->
        <geom name="wall_{i}" type="mesh" mesh="wall_{i}" pos="0 0 0" 
              rgba="0.8 0.8 0.9 1" 
              contype="1" conaffinity="1" 
              friction="1 0.5 0.5"/>
"""
        
        # Add environment mesh
        if 'environment' in self.mesh_files:
            xml += """        
        <!-- Environment mesh (building) -->
        <geom name="environment" type="mesh" mesh="environment" pos="0 0 0" 
              rgba="0.7 0.7 0.7 1" 
              contype="1" conaffinity="1"
              friction="1 0.5 0.5"/>
"""
        # Add humanoid agent
        xml += f"""
        <!-- Humanoid Agent -->
        <body name="torso" pos="0 0 {agent_start_z}">
            <freejoint name="root"/>
            <geom name="torso" type="capsule" size="0.07" fromto="0 0 -0.28 0 0 0.28" 
                  mass="10" rgba="0.8 0.6 0.4 1"/>
            
            <!-- Head -->
            <body name="head" pos="0 0 0.4">
                <geom name="head" type="sphere" size="0.09" mass="2" rgba="1 0.8 0.6 1"/>
            </body>
            
            <!-- Left Arm -->
            <body name="left_upper_arm" pos="0 0.18 0.2">
                <joint name="left_shoulder_x" type="hinge" axis="1 0 0" range="-85 60"/>
                <joint name="left_shoulder_y" type="hinge" axis="0 1 0" range="-85 85"/>
                <geom name="left_upper_arm" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" 
                      mass="2" rgba="0.8 0.6 0.4 1"/>
                
                <body name="left_lower_arm" pos="0 0 -0.3">
                    <joint name="left_elbow" type="hinge" axis="0 1 0" range="-90 50"/>
                    <geom name="left_lower_arm" type="capsule" size="0.031" fromto="0 0 0 0 0 -0.25" 
                          mass="1" rgba="0.8 0.6 0.4 1"/>
                    
                    <body name="left_hand" pos="0 0 -0.25">
                        <geom name="left_hand" type="sphere" size="0.04" mass="0.3" rgba="1 0.8 0.6 1"/>
                    </body>
                </body>
            </body>
            
            <!-- Right Arm -->
            <body name="right_upper_arm" pos="0 -0.18 0.2">
                <joint name="right_shoulder_x" type="hinge" axis="1 0 0" range="-85 60"/>
                <joint name="right_shoulder_y" type="hinge" axis="0 1 0" range="-85 85"/>
                <geom name="right_upper_arm" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.3" 
                      mass="2" rgba="0.8 0.6 0.4 1"/>
                
                <body name="right_lower_arm" pos="0 0 -0.3">
                    <joint name="right_elbow" type="hinge" axis="0 1 0" range="-90 50"/>
                    <geom name="right_lower_arm" type="capsule" size="0.031" fromto="0 0 0 0 0 -0.25" 
                          mass="1" rgba="0.8 0.6 0.4 1"/>
                    
                    <body name="right_hand" pos="0 0 -0.25">
                        <geom name="right_hand" type="sphere" size="0.04" mass="0.3" rgba="1 0.8 0.6 1"/>
                    </body>
                </body>
            </body>
            
            <!-- Left Leg -->
            <body name="left_thigh" pos="0 0.1 -0.28">
                <joint name="left_hip_x" type="hinge" axis="1 0 0" range="-25 5"/>
                <joint name="left_hip_y" type="hinge" axis="0 1 0" range="-25 25"/>
                <joint name="left_hip_z" type="hinge" axis="0 0 1" range="-60 35"/>
                <geom name="left_thigh" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.45" 
                      mass="4" rgba="0.8 0.6 0.4 1"/>
                
                <body name="left_shin" pos="0 0 -0.45">
                    <joint name="left_knee" type="hinge" axis="0 1 0" range="-160 -2"/>
                    <geom name="left_shin" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.4" 
                          mass="2.5" rgba="0.8 0.6 0.4 1"/>
                    
                    <body name="left_foot" pos="0 0 -0.4">
                        <joint name="left_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
                        <joint name="left_ankle_x" type="hinge" axis="1 0 0" range="-50 50"/>
                        <geom name="left_foot" type="box" size="0.13 0.05 0.02" 
                              mass="1" rgba="0.9 0.7 0.5 1"/>
                    </body>
                </body>
            </body>
            
            <!-- Right Leg -->
            <body name="right_thigh" pos="0 -0.1 -0.28">
                <joint name="right_hip_x" type="hinge" axis="1 0 0" range="-25 5"/>
                <joint name="right_hip_y" type="hinge" axis="0 1 0" range="-25 25"/>
                <joint name="right_hip_z" type="hinge" axis="0 0 1" range="-35 60"/>
                <geom name="right_thigh" type="capsule" size="0.05" fromto="0 0 0 0 0 -0.45" 
                      mass="4" rgba="0.8 0.6 0.4 1"/>
                
                <body name="right_shin" pos="0 0 -0.45">
                    <joint name="right_knee" type="hinge" axis="0 1 0" range="-160 -2"/>
                    <geom name="right_shin" type="capsule" size="0.04" fromto="0 0 0 0 0 -0.4" 
                          mass="2.5" rgba="0.8 0.6 0.4 1"/>
                    
                    <body name="right_foot" pos="0 0 -0.4">
                        <joint name="right_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
                        <joint name="right_ankle_x" type="hinge" axis="1 0 0" range="-50 50"/>
                        <geom name="right_foot" type="box" size="0.13 0.05 0.02" 
                              mass="1" rgba="0.9 0.7 0.5 1"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <!-- Arm motors -->
        <motor name="left_shoulder_x" joint="left_shoulder_x" gear="25"/>
        <motor name="left_shoulder_y" joint="left_shoulder_y" gear="25"/>
        <motor name="left_elbow" joint="left_elbow" gear="25"/>
        <motor name="right_shoulder_x" joint="right_shoulder_x" gear="25"/>
        <motor name="right_shoulder_y" joint="right_shoulder_y" gear="25"/>
        <motor name="right_elbow" joint="right_elbow" gear="25"/>
        
        <!-- Leg motors -->
        <motor name="left_hip_x" joint="left_hip_x" gear="100"/>
        <motor name="left_hip_y" joint="left_hip_y" gear="100"/>
        <motor name="left_hip_z" joint="left_hip_z" gear="100"/>
        <motor name="left_knee" joint="left_knee" gear="100"/>
        <motor name="left_ankle_y" joint="left_ankle_y" gear="50"/>
        <motor name="left_ankle_x" joint="left_ankle_x" gear="50"/>
        
        <motor name="right_hip_x" joint="right_hip_x" gear="100"/>
        <motor name="right_hip_y" joint="right_hip_y" gear="100"/>
        <motor name="right_hip_z" joint="right_hip_z" gear="100"/>
        <motor name="right_knee" joint="right_knee" gear="100"/>
        <motor name="right_ankle_y" joint="right_ankle_y" gear="50"/>
        <motor name="right_ankle_x" joint="right_ankle_x" gear="50"/>
    </actuator>
</mujoco>
"""
        
        return xml


class HumanoidWalkingEnv:
    """MuJoCo Humanoid Walking Environment"""
    
    def __init__(self, model_xml):
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.action_dim = self.model.nu  # Number of actuators
        self.obs_dim = self._get_observation().shape[0]
        
        # Episode tracking
        self.episode_length = 1000
        self.step_count = 0
        
        # Reference positions for initial pose
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        
    def _get_observation(self):
        """Get current observation state"""
        # Get positions and velocities
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Get center of mass
        com_pos = self.data.subtree_com[1].copy()  # torso com
        com_vel = self.data.cvel[1].copy()[:3]     # linear velocity
        
        # Get torso orientation
        torso_quat = qpos[3:7]
        
        # Get foot contact forces
        left_foot_contact = self._get_contact_force("left_foot")
        right_foot_contact = self._get_contact_force("right_foot")
        
        # Combine observations
        obs = np.concatenate([
            qpos[7:],  # joint positions (excluding root translation/rotation)
            qvel,      # all velocities
            com_pos,   # center of mass position
            com_vel,   # center of mass velocity
            torso_quat, # torso orientation
            [left_foot_contact, right_foot_contact]  # foot contacts
        ])
        
        return obs
    
    def _get_contact_force(self, geom_name):
        """Get contact force magnitude for a geom"""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        contact_force = 0.0
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == geom_id or contact.geom2 == geom_id:
                # Get contact force
                c_array = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                contact_force += np.linalg.norm(c_array[:3])
        
        return min(contact_force / 1000.0, 1.0)  # Normalize
    
    def reset(self):
        """Reset environment to initial state"""
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        
        # Add small random perturbation
        self.data.qpos[7:] += np.random.normal(0, 0.02, size=len(self.data.qpos[7:]))
        self.data.qvel[:] += np.random.normal(0, 0.01, size=len(self.data.qvel))
        
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        
        return self._get_observation()
    
    def step(self, action):
        """Take environment step"""
        # Apply action
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self._is_terminal()
        
        self.step_count += 1
        if self.step_count >= self.episode_length:
            done = True
        
        return obs, reward, done, {}
    
    def _calculate_reward(self):
        """Calculate walking reward"""
        # Forward velocity reward
        forward_vel = self.data.cvel[1][0]  # torso forward velocity
        forward_reward = min(forward_vel, 2.0)  # Cap at 2 m/s
        
        # Height penalty (want to stay upright)
        torso_height = self.data.qpos[2]
        height_reward = -abs(torso_height - 1.3)  # Target height around 1.3m
        
        # Orientation penalty (stay upright)
        torso_quat = self.data.qpos[3:7]
        up_vec = np.array([0, 0, 1])
        torso_up = self._quat_to_up_vector(torso_quat)
        upright_reward = np.dot(torso_up, up_vec) - 1.0
        
        # Energy penalty (smooth movement)
        action_penalty = -0.01 * np.sum(self.data.ctrl ** 2)
        
        # Contact reward (encourage ground contact)
        left_contact = self._get_contact_force("left_foot")
        right_contact = self._get_contact_force("right_foot")
        contact_reward = 0.1 * (left_contact + right_contact)
        
        # Total reward
        total_reward = (forward_reward + 
                       0.5 * height_reward + 
                       0.5 * upright_reward + 
                       action_penalty + 
                       contact_reward)
        
        return total_reward
    
    def _quat_to_up_vector(self, quat):
        """Convert quaternion to up vector"""
        w, x, y, z = quat
        up = np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y)
        ])
        return up / np.linalg.norm(up)
    
    def _is_terminal(self):
        """Check if episode should terminate"""
        # Terminate if fallen (torso too low)
        if self.data.qpos[2] < 0.5:
            return True
        
        # Terminate if torso tilted too much
        torso_quat = self.data.qpos[3:7]
        up_vec = np.array([0, 0, 1])
        torso_up = self._quat_to_up_vector(torso_quat)
        if np.dot(torso_up, up_vec) < 0.3:  # More than ~70 degrees tilt
            return True
        
        return False

