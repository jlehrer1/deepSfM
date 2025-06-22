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
    
    def run_drop_test(self, model, data, duration=5.0, log_interval=1.0):
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
        
        # Final analysis
        final_z = positions[-1][1]
        print(f"\nDrop test results:")
        print(f"  Started at: Z={agent_start_z:.3f}")
        print(f"  Ended at: Z={final_z:.3f}")
        print(f"  Expected: Z={expected_z:.3f} (ground + radius)")
        print(f"  Ground level: Z={ground_z:.3f}")
        
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
        
        # Load in MuJoCo
        try:
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            print("✅ Successfully loaded in MuJoCo!")
        except Exception as e:
            print(f"❌ Failed to load in MuJoCo: {e}")
            return False
        
        # Run drop test
        success, positions = self.run_drop_test(model, data)
        
        # Interactive visualization
        if visualize:
            print("\nLaunching interactive viewer...")
            print("Controls:")
            print("  - Close window to exit")
            print("  - Ball resets every 5 seconds")
            
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
        
        return success


def test_sfm_environment(data_path='camera_view_ground_env.npz', 
                        enable_env_collision=True,
                        visualize_meshes=False,
                        test_physics=True):
    """
    Complete pipeline to test SfM environment in MuJoCo.
    
    Args:
        data_path: Path to normalized SfM data (.npz file)
        enable_env_collision: Whether building mesh has collision
        visualize_meshes: Show meshes in Open3D before testing
        test_physics: Run MuJoCo physics test
    """
    print("="*60)
    print("SfM to MuJoCo Environment Test")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
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
    test_sfm_environment(enable_env_collision=True)