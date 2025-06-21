import numpy as np
import open3d as o3d
import mujoco
import mujoco.viewer
import time
from deepsfm.mesh_creator import create_fixed_rl_environment

def add_thickness_to_mesh(mesh, thickness=0.1):
    """Add thickness to thin mesh surfaces to make them solid"""
    
    print(f"Adding {thickness} thickness to mesh...")
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Compute face normals
    mesh.compute_triangle_normals()
    triangle_normals = np.asarray(mesh.triangle_normals)
    
    # For each triangle, create a thick version by extruding along normal
    new_vertices = []
    new_triangles = []
    vertex_count = 0
    
    for i, (tri, normal) in enumerate(zip(triangles, triangle_normals)):
        # Get original triangle vertices
        v1, v2, v3 = vertices[tri]
        
        # Create extruded vertices (offset along normal)
        v1_ext = v1 - normal * thickness
        v2_ext = v2 - normal * thickness  
        v3_ext = v3 - normal * thickness
        
        # Add all 6 vertices (original + extruded)
        new_vertices.extend([v1, v2, v3, v1_ext, v2_ext, v3_ext])
        
        # Create triangles for the thick surface
        base = vertex_count
        
        # Original face
        new_triangles.append([base, base+1, base+2])
        # Extruded face (flipped normal)
        new_triangles.append([base+3, base+5, base+4])
        
        # Side faces connecting original to extruded
        # Side 1 (v1-v2 edge)
        new_triangles.extend([
            [base, base+3, base+4], [base, base+4, base+1]
        ])
        # Side 2 (v2-v3 edge)  
        new_triangles.extend([
            [base+1, base+4, base+5], [base+1, base+5, base+2]
        ])
        # Side 3 (v3-v1 edge)
        new_triangles.extend([
            [base+2, base+5, base+3], [base+2, base+3, base]
        ])
        
        vertex_count += 6
    
    # Create new mesh
    thick_mesh = o3d.geometry.TriangleMesh()
    thick_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    thick_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    # Clean up
    thick_mesh = thick_mesh.remove_duplicated_vertices()
    thick_mesh = thick_mesh.remove_duplicated_triangles()
    thick_mesh = thick_mesh.remove_degenerate_triangles()
    thick_mesh.compute_vertex_normals()
    
    print(f"Original: {len(vertices)} vertices, {len(triangles)} triangles")
    print(f"Thick: {len(thick_mesh.vertices)} vertices, {len(thick_mesh.triangles)} triangles")
    
    return thick_mesh

def create_thick_ground_and_walls(normalized_data, wall_thickness=0.2, ground_thickness=0.3):
    """Create thick, solid ground and walls from your environment bounds"""
    
    points = normalized_data['points']
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    print(f"Environment bounds: {min_bound} to {max_bound}")
    
    meshes = []
    
    # 1. Thick ground plane
    ground_size_x = max_bound[0] - min_bound[0] + 2  # Add margin
    ground_size_y = max_bound[1] - min_bound[1] + 2
    ground_z = min_bound[2]
    
    ground = o3d.geometry.TriangleMesh.create_box(
        ground_size_x, ground_size_y, ground_thickness
    )
    ground.translate([
        min_bound[0] - 1,
        min_bound[1] - 1, 
        ground_z - ground_thickness
    ])
    ground.paint_uniform_color([0.6, 0.4, 0.2])
    meshes.append(ground)
    
    # 2. Thick boundary walls
    wall_height = max_bound[2] - min_bound[2] + 1
    margin = 1.5
    
    # Front wall
    wall = o3d.geometry.TriangleMesh.create_box(
        ground_size_x + 2*wall_thickness, wall_thickness, wall_height
    )
    wall.translate([
        min_bound[0] - 1 - wall_thickness,
        min_bound[1] - 1 - wall_thickness,
        min_bound[2]
    ])
    wall.paint_uniform_color([0.8, 0.8, 0.9])
    meshes.append(wall)
    
    # Back wall
    wall = o3d.geometry.TriangleMesh.create_box(
        ground_size_x + 2*wall_thickness, wall_thickness, wall_height
    )
    wall.translate([
        min_bound[0] - 1 - wall_thickness,
        max_bound[1] + 1,
        min_bound[2]
    ])
    wall.paint_uniform_color([0.8, 0.8, 0.9])
    meshes.append(wall)
    
    # Left wall
    wall = o3d.geometry.TriangleMesh.create_box(
        wall_thickness, ground_size_y, wall_height
    )
    wall.translate([
        min_bound[0] - 1 - wall_thickness,
        min_bound[1] - 1,
        min_bound[2]
    ])
    wall.paint_uniform_color([0.8, 0.8, 0.9])
    meshes.append(wall)
    
    # Right wall
    wall = o3d.geometry.TriangleMesh.create_box(
        wall_thickness, ground_size_y, wall_height
    )
    wall.translate([
        max_bound[0] + 1,
        min_bound[1] - 1,
        min_bound[2]
    ])
    wall.paint_uniform_color([0.8, 0.8, 0.9])
    meshes.append(wall)
    
    # Combine all
    combined = meshes[0]
    for mesh in meshes[1:]:
        combined += mesh
    
    combined.remove_duplicated_vertices()
    combined.remove_duplicated_triangles()
    combined.compute_vertex_normals()
    
    print(f"Created thick environment: {len(combined.vertices)} vertices, {len(combined.triangles)} triangles")
    
    return combined

def fix_your_original_environment():
    """Fix your original environment to work with mesh collision"""
    
    print("=== FIXING YOUR ORIGINAL ENVIRONMENT ===")
    
    # Load your data
    data = np.load('camera_view_ground_env.npz')
    normalized_data = {
        'points': data['points'],
        'colors': data['colors'],
        'camera_positions': data['camera_positions'],
        'camera_directions': data['camera_directions']
    }
    
    print("Loading your original environment...")
    
    # Method 1: Try to fix your existing mesh by adding thickness
    try:
        rl_env = create_fixed_rl_environment(
            normalized_data, add_walls=True, visualize=False
        )
        
        original_mesh = rl_env['environment_mesh'] + rl_env['ground_mesh']
        if rl_env['walls']:
            for wall in rl_env['walls']:
                original_mesh += wall
        
        print("Attempting to add thickness to your original mesh...")
        thick_mesh = add_thickness_to_mesh(original_mesh, thickness=0.15)
        
        # Test this mesh
        test_mesh_collision(thick_mesh, "thick_original", normalized_data)
        
    except Exception as e:
        print(f"Failed to thicken original mesh: {e}")
        thick_mesh = None
    
    # Method 2: Create thick ground and walls from scratch
    print("\nCreating thick ground and walls from environment bounds...")
    simple_thick_mesh = create_thick_ground_and_walls(normalized_data)
    
    # Test this mesh
    test_mesh_collision(simple_thick_mesh, "thick_simple", normalized_data)
    
    return thick_mesh, simple_thick_mesh

def test_mesh_collision(mesh, name, normalized_data):
    """Test a mesh for collision in MuJoCo"""
    
    print(f"\n--- Testing {name} mesh ---")
    
    # Save mesh
    filename = f"{name}_environment.obj"
    success = o3d.io.write_triangle_mesh(filename, mesh, write_ascii=True)
    
    if not success:
        print(f"❌ Failed to save {filename}")
        return False
    
    # Get bounds
    bounds = mesh.get_axis_aligned_bounding_box()
    min_bound = bounds.min_bound
    max_bound = bounds.max_bound
    agent_start = max_bound[2] + 1.0
    
    # Create MuJoCo XML
    mjcf = f"""
<mujoco model="{name}_test">
    <compiler angle="degree" meshdir="." />
    <option timestep="0.01" gravity="0 0 -9.81"/>
    
    <asset>
        <mesh name="env_mesh" file="{filename}"/>
    </asset>
    
    <worldbody>
        <!-- Your fixed mesh -->
        <geom name="environment" type="mesh" mesh="env_mesh" pos="0 0 0" 
              rgba="0.7 0.7 0.7 1" 
              contype="1" conaffinity="1" 
              friction="1 0.5 0.5"/>
        
        <!-- Test ball -->
        <body name="ball" pos="0 0 {agent_start}">
            <freejoint/>
            <geom name="ball_geom" type="sphere" size="0.15" 
                  rgba="1 0.2 0.2 1"
                  contype="2" conaffinity="1"
                  mass="1.0"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.06 0.06 0.06"/>
        </body>
        
        <!-- Camera positions for reference -->
"""
    
    # Add camera position markers
    for i, pos in enumerate(normalized_data['camera_positions'][:5]):  # First 5 cameras
        mjcf += f"""        <geom name="cam_{i}" type="sphere" size="0.05" pos="{pos[0]} {pos[1]} {pos[2]}" 
              rgba="0 1 0 0.7" contype="0" conaffinity="0"/>
"""
    
    mjcf += f"""    </worldbody>
    
    <contact>
        <pair geom1="ball_geom" geom2="environment"/>
    </contact>
</mujoco>
"""
    
    # Test the model
    try:
        model = mujoco.MjModel.from_xml_string(mjcf)
        data = mujoco.MjData(model)
        
        print(f"✅ {name} model loaded successfully!")
        
        # Run physics test
        for step in range(400):
            mujoco.mj_step(model, data)
            
            if step in [100, 200, 300, 399]:
                ball_z = data.qpos[2]
                fall_distance = agent_start - ball_z
                print(f"  t={step/100:.1f}s: Ball Z={ball_z:.3f}, fell={fall_distance:.3f}")
        
        final_z = data.qpos[2]
        expected_ground = min_bound[2] + 0.15  # Ball radius
        
        if abs(final_z - expected_ground) < 0.5:
            print(f"🎉 {name} SUCCESS! Ball landed at Z={final_z:.3f}")
            
            # Launch interactive viewer for successful mesh
            print(f"Launching interactive viewer for {name}...")
            with mujoco.viewer.launch(model, data) as viewer:
                step = 0
                while viewer.is_running() and step < 1000:  # Run for 10 seconds
                    if step % 500 == 0 and step > 0:  # Reset every 5 seconds
                        data.qpos[0:3] = [0, 0, agent_start]
                        data.qpos[3:7] = [1, 0, 0, 0]
                        data.qvel[:] = 0
                        mujoco.mj_forward(model, data)
                    
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    step += 1
                    time.sleep(0.01)
            
            return True
        else:
            print(f"❌ {name} FAILED: Ball at Z={final_z:.3f}, expected ~{expected_ground:.3f}")
            return False
    
    except Exception as e:
        print(f"❌ {name} ERROR: {e}")
        return False

if __name__ == "__main__":
    fix_your_original_environment()