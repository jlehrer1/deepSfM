import numpy as np
import open3d as o3d
import mujoco
import mujoco.viewer
import time

# Load your data
data = np.load('camera_view_ground_env.npz')
normalized_data = {
    'points': data['points'],
    'colors': data['colors'],
    'camera_positions': data['camera_positions'],
    'camera_directions': data['camera_directions']
}

def debug_mesh_properties(mesh, name="mesh"):
    """Debug mesh properties that might cause collision issues"""
    print(f"\n=== {name} Debug Info ===")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.triangles)}")
    
    # Check mesh integrity
    print(f"Watertight: {mesh.is_watertight()}")
    print(f"Vertex manifold: {mesh.is_vertex_manifold()}")
    print(f"Edge manifold: {mesh.is_edge_manifold()}")
    print(f"Self-intersecting: {mesh.is_self_intersecting()}")
    
    # Check bounds
    bounds = mesh.get_axis_aligned_bounding_box()
    print(f"Bounds: min={bounds.min_bound}, max={bounds.max_bound}")
    
    # Check normals
    if len(mesh.triangle_normals) > 0:
        normals = np.asarray(mesh.triangle_normals)
        up_facing = np.sum(normals[:, 2] > 0.8)  # Nearly vertical up
        down_facing = np.sum(normals[:, 2] < -0.8)  # Nearly vertical down
        print(f"Up-facing triangles: {up_facing}")
        print(f"Down-facing triangles: {down_facing}")
        
        # Check for horizontal faces (potential ceiling)
        horizontal = np.sum(np.abs(normals[:, 2]) > 0.9)
        print(f"Horizontal faces (potential ceiling): {horizontal}")

def create_simple_test_environment():
    """Create the simplest possible environment to test"""
    print("Creating minimal test environment...")
    
    # Just a ground plane and 4 walls - nothing fancy
    ground_size = 10
    wall_height = 3
    wall_thickness = 0.2
    
    # Ground plane
    ground = o3d.geometry.TriangleMesh.create_box(ground_size, ground_size, 0.1)
    ground.translate([-ground_size/2, -ground_size/2, -0.1])
    ground.paint_uniform_color([0.6, 0.4, 0.2])
    
    # 4 walls (NO TOP)
    walls = []
    
    # Front wall
    wall = o3d.geometry.TriangleMesh.create_box(ground_size, wall_thickness, wall_height)
    wall.translate([-ground_size/2, -ground_size/2 - wall_thickness, 0])
    walls.append(wall)
    
    # Back wall
    wall = o3d.geometry.TriangleMesh.create_box(ground_size, wall_thickness, wall_height)
    wall.translate([-ground_size/2, ground_size/2, 0])
    walls.append(wall)
    
    # Left wall
    wall = o3d.geometry.TriangleMesh.create_box(wall_thickness, ground_size, wall_height)
    wall.translate([-ground_size/2 - wall_thickness, -ground_size/2, 0])
    walls.append(wall)
    
    # Right wall
    wall = o3d.geometry.TriangleMesh.create_box(wall_thickness, ground_size, wall_height)
    wall.translate([ground_size/2, -ground_size/2, 0])
    walls.append(wall)
    
    # Combine all
    combined = ground
    for wall in walls:
        wall.paint_uniform_color([0.8, 0.8, 0.9])
        combined += wall
    
    combined.compute_vertex_normals()
    
    return combined

def create_point_cloud_only_environment():
    """Use ONLY the point cloud data, no mesh reconstruction"""
    print("Creating environment from point cloud only...")
    
    points = normalized_data['points']
    min_z = points[:, 2].min()
    max_z = points[:, 2].max()
    
    # Create ground at minimum Z
    ground_size = 15
    ground_thickness = 0.2
    ground = o3d.geometry.TriangleMesh.create_box(ground_size, ground_size, ground_thickness)
    ground.translate([-ground_size/2, -ground_size/2, min_z - ground_thickness])
    ground.paint_uniform_color([0.6, 0.4, 0.2])
    
    # Create walls around the point cloud bounds
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    
    wall_height = max_z - min_z + 1
    wall_thickness = 0.3
    
    # Expand bounds slightly
    margin = 1.0
    x_min, y_min = bounds_min[:2] - margin
    x_max, y_max = bounds_max[:2] + margin
    
    walls = []
    
    # 4 boundary walls
    wall_configs = [
        ([x_min - wall_thickness, y_min, min_z], [wall_thickness, y_max - y_min, wall_height]),
        ([x_max, y_min, min_z], [wall_thickness, y_max - y_min, wall_height]),
        ([x_min, y_min - wall_thickness, min_z], [x_max - x_min, wall_thickness, wall_height]),
        ([x_min, y_max, min_z], [x_max - x_min, wall_thickness, wall_height])
    ]
    
    for pos, size in wall_configs:
        wall = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2])
        wall.translate(pos)
        wall.paint_uniform_color([0.8, 0.8, 0.9])
        walls.append(wall)
    
    # Combine
    combined = ground
    for wall in walls:
        combined += wall
    
    combined.remove_duplicated_vertices()
    combined.remove_duplicated_triangles()
    combined.compute_vertex_normals()
    
    return combined

def test_environment_options():
    """Test different environment creation approaches"""
    
    print("\n" + "="*60)
    print("TESTING DIFFERENT ENVIRONMENT APPROACHES")
    print("="*60)
    
    # Option 1: Simple test environment
    print("\n1. Testing simple box environment...")
    simple_env = create_simple_test_environment()
    debug_mesh_properties(simple_env, "Simple Environment")
    test_mesh_in_mujoco(simple_env, "simple_test.obj", "Simple Test")
    
    # Option 2: Point cloud bounds only
    print("\n2. Testing point cloud bounds environment...")
    pc_env = create_point_cloud_only_environment()
    debug_mesh_properties(pc_env, "Point Cloud Environment")
    test_mesh_in_mujoco(pc_env, "pointcloud_test.obj", "Point Cloud Test")

def test_mesh_in_mujoco(mesh, filename, description):
    """Test a mesh in MuJoCo to see if ball falls properly"""
    
    # Save mesh
    success = o3d.io.write_triangle_mesh(filename, mesh, write_ascii=True)
    if not success:
        print(f"Failed to save {filename}")
        return
    
    bounds = mesh.get_axis_aligned_bounding_box()
    min_bound = bounds.min_bound
    max_bound = bounds.max_bound
    
    agent_start_height = max_bound[2] + 2.0
    ground_level = min_bound[2]
    
    # Create MuJoCo XML
    mjcf = f"""
<mujoco model="test_{description.lower().replace(' ', '_')}">
    <compiler angle="degree" meshdir="." />
    <option timestep="0.01" gravity="0 0 -9.81"/>
    
    <asset>
        <mesh name="test_mesh" file="{filename}"/>
    </asset>
    
    <worldbody>
        <!-- Test environment -->
        <geom name="environment" type="mesh" mesh="test_mesh" pos="0 0 0" 
              rgba="0.7 0.7 0.7 1" 
              contype="1" conaffinity="1" 
              friction="1 0.5 0.5"/>
        
        <!-- Test ball -->
        <body name="agent" pos="0 0 {agent_start_height}">
            <freejoint/>
            <geom name="agent_body" type="sphere" size="0.2" 
                  rgba="1 0 0 1"
                  contype="2" conaffinity="1"
                  mass="1.0"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.08 0.08 0.08"/>
        </body>
        
        <!-- Visual reference -->
        <geom name="reference" type="sphere" size="0.1" pos="0 0 {max_bound[2] + 0.5}"
              rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    </worldbody>
    
    <contact>
        <pair geom1="agent_body" geom2="environment"/>
    </contact>
</mujoco>
"""
    
    print(f"\nTesting {description}:")
    print(f"Agent starts at Z = {agent_start_height:.2f}")
    print(f"Ground at Z = {ground_level:.2f}")
    print(f"Expected fall distance = {agent_start_height - ground_level:.2f}")
    
    try:
        model = mujoco.MjModel.from_xml_string(mjcf)
        data = mujoco.MjData(model)
        
        # Run a quick simulation test
        print("Running 5-second physics test...")
        
        for step in range(500):  # 5 seconds
            mujoco.mj_step(model, data)
            
            if step % 100 == 0:  # Every second
                agent_z = data.qpos[2]
                fall_distance = agent_start_height - agent_z
                print(f"  t={step/100:.0f}s: Z={agent_z:.3f}, fell {fall_distance:.3f}")
        
        final_z = data.qpos[2]
        total_fall = agent_start_height - final_z
        
        if total_fall < 0.5:
            print(f"❌ PROBLEM: Ball only fell {total_fall:.3f} units - likely hitting invisible collision!")
        elif final_z > ground_level + 0.5:
            print(f"❌ PROBLEM: Ball stopped at Z={final_z:.3f}, should be near {ground_level:.3f}")
        else:
            print(f"✅ SUCCESS: Ball fell {total_fall:.3f} units and landed properly")
            
        return model, data
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None

# Run the tests
if __name__ == "__main__":
    test_environment_options()
    
    print("\n" + "="*60)
    print("If the simple test works but point cloud test fails,")
    print("the issue is with mesh reconstruction from your point cloud.")
    print("If both fail, it's a more fundamental MuJoCo setup issue.")
    print("="*60)