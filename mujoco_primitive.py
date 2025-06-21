import numpy as np
import open3d as o3d
import mujoco
import mujoco.viewer
import time

def completely_remove_horizontal_faces(mesh, z_threshold=0.9):
    """Remove ALL horizontal faces from mesh"""
    
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    triangles = np.asarray(mesh.triangles)
    
    # Keep only triangles that are NOT horizontal
    # (normal Z component less than threshold)
    keep_mask = np.abs(normals[:, 2]) < z_threshold
    filtered_triangles = triangles[keep_mask]
    
    print(f"Removed {len(triangles) - len(filtered_triangles)} horizontal faces")
    print(f"Kept {len(filtered_triangles)} non-horizontal faces")
    
    # Create new mesh
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = mesh.vertices
    new_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)
    new_mesh.vertex_colors = mesh.vertex_colors
    new_mesh.compute_vertex_normals()
    
    return new_mesh

def create_ultra_clean_environment():
    """Create environment with absolutely no horizontal faces except ground"""
    
    ground_size = 10
    wall_height = 3
    wall_thickness = 0.2
    
    meshes = []
    
    # 1. Ground plane (the ONLY horizontal surface)
    ground_vertices = np.array([
        [-ground_size/2, -ground_size/2, 0],
        [ground_size/2, -ground_size/2, 0],
        [ground_size/2, ground_size/2, 0], 
        [-ground_size/2, ground_size/2, 0]
    ])
    
    ground_triangles = np.array([[0, 1, 2], [0, 2, 3]])
    
    ground = o3d.geometry.TriangleMesh()
    ground.vertices = o3d.utility.Vector3dVector(ground_vertices)
    ground.triangles = o3d.utility.Vector3iVector(ground_triangles)
    ground.paint_uniform_color([0.6, 0.4, 0.2])
    meshes.append(ground)
    
    # 2. Create 4 walls as separate vertical rectangles (NO horizontal faces)
    wall_configs = [
        # Front wall
        {
            'vertices': np.array([
                [-ground_size/2, -ground_size/2, 0],
                [ground_size/2, -ground_size/2, 0], 
                [ground_size/2, -ground_size/2, wall_height],
                [-ground_size/2, -ground_size/2, wall_height]
            ]),
            'triangles': np.array([[0, 1, 2], [0, 2, 3]])
        },
        # Back wall  
        {
            'vertices': np.array([
                [ground_size/2, ground_size/2, 0],
                [-ground_size/2, ground_size/2, 0],
                [-ground_size/2, ground_size/2, wall_height], 
                [ground_size/2, ground_size/2, wall_height]
            ]),
            'triangles': np.array([[0, 1, 2], [0, 2, 3]])
        },
        # Left wall
        {
            'vertices': np.array([
                [-ground_size/2, ground_size/2, 0],
                [-ground_size/2, -ground_size/2, 0],
                [-ground_size/2, -ground_size/2, wall_height],
                [-ground_size/2, ground_size/2, wall_height] 
            ]),
            'triangles': np.array([[0, 1, 2], [0, 2, 3]])
        },
        # Right wall
        {
            'vertices': np.array([
                [ground_size/2, -ground_size/2, 0],
                [ground_size/2, ground_size/2, 0], 
                [ground_size/2, ground_size/2, wall_height],
                [ground_size/2, -ground_size/2, wall_height]
            ]),
            'triangles': np.array([[0, 1, 2], [0, 2, 3]])
        }
    ]
    
    for config in wall_configs:
        wall = o3d.geometry.TriangleMesh()
        wall.vertices = o3d.utility.Vector3dVector(config['vertices'])
        wall.triangles = o3d.utility.Vector3iVector(config['triangles']) 
        wall.paint_uniform_color([0.8, 0.8, 0.9])
        meshes.append(wall)
    
    # Combine all
    combined = meshes[0]
    for mesh in meshes[1:]:
        combined += mesh
    
    combined.remove_duplicated_vertices()
    combined.remove_duplicated_triangles()
    combined.compute_vertex_normals()
    
    return combined

def test_primitive_collision_alternative():
    """Alternative: Use MuJoCo primitives for collision, mesh for visual"""
    
    ground_size = 10
    wall_height = 3
    wall_thickness = 0.2
    agent_start = 5.0
    
    mjcf = f"""
<mujoco model="primitive_collision">
    <compiler angle="degree" meshdir="." />
    <option timestep="0.01" gravity="0 0 -9.81"/>
    
    <worldbody>
        <!-- Use primitive geometry for collision (guaranteed no tops) -->
        <geom name="ground_collision" type="box" 
              size="{ground_size/2} {ground_size/2} 0.05" 
              pos="0 0 -0.05" 
              rgba="0.6 0.4 0.2 1"
              contype="1" conaffinity="1" 
              friction="1 0.5 0.5"/>
        
        <!-- 4 wall collision boxes -->
        <geom name="wall_front" type="box" 
              size="{ground_size/2} {wall_thickness/2} {wall_height/2}" 
              pos="0 {-ground_size/2 - wall_thickness/2} {wall_height/2}" 
              rgba="0.8 0.8 0.9 1"
              contype="1" conaffinity="1"/>
              
        <geom name="wall_back" type="box" 
              size="{ground_size/2} {wall_thickness/2} {wall_height/2}" 
              pos="0 {ground_size/2 + wall_thickness/2} {wall_height/2}" 
              rgba="0.8 0.8 0.9 1"
              contype="1" conaffinity="1"/>
              
        <geom name="wall_left" type="box" 
              size="{wall_thickness/2} {ground_size/2} {wall_height/2}" 
              pos="{-ground_size/2 - wall_thickness/2} 0 {wall_height/2}" 
              rgba="0.8 0.8 0.9 1"
              contype="1" conaffinity="1"/>
              
        <geom name="wall_right" type="box" 
              size="{wall_thickness/2} {ground_size/2} {wall_height/2}" 
              pos="{ground_size/2 + wall_thickness/2} 0 {wall_height/2}" 
              rgba="0.8 0.8 0.9 1"
              contype="1" conaffinity="1"/>
        
        <!-- Test agent -->
        <body name="agent" pos="0 0 {agent_start}">
            <freejoint/>
            <geom name="agent_body" type="sphere" size="0.2" 
                  rgba="1 0 0 1"
                  contype="2" conaffinity="1"
                  mass="1.0"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.08 0.08 0.08"/>
        </body>
    </worldbody>
    
    <contact>
        <pair geom1="agent_body" geom2="ground_collision"/>
        <pair geom1="agent_body" geom2="wall_front"/>
        <pair geom1="agent_body" geom2="wall_back"/>
        <pair geom1="agent_body" geom2="wall_left"/>
        <pair geom1="agent_body" geom2="wall_right"/>
    </contact>
</mujoco>
"""
    
    print("Testing primitive collision alternative...")
    print("This uses MuJoCo box primitives - guaranteed no invisible tops!")
    
    try:
        model = mujoco.MjModel.from_xml_string(mjcf)
        data = mujoco.MjData(model)
        
        print("✅ Primitive model loaded!")
        
        # Test physics
        for step in range(500):
            mujoco.mj_step(model, data)
            
            if step % 100 == 0:
                agent_z = data.qpos[2]
                fall_distance = agent_start - agent_z
                velocity = data.qvel[2]
                print(f"t={step/100:.0f}s: Z={agent_z:.3f}, fell={fall_distance:.3f}, vz={velocity:.3f}")
        
        final_z = data.qpos[2]
        expected_final = 0.2  # Ball radius above ground
        
        if abs(final_z - expected_final) < 0.3:
            print(f"✅ PRIMITIVE SUCCESS! Ball at Z={final_z:.3f}")
            return True
        else:
            print(f"❌ Primitive failed: Ball at Z={final_z:.3f}")
            return False
            
    except Exception as e:
        print(f"❌ Primitive test failed: {e}")
        return False

def test_ultra_clean_mesh():
    """Test the ultra-clean mesh with no horizontal faces"""
    
    print("Creating ultra-clean mesh (only ground is horizontal)...")
    mesh = create_ultra_clean_environment()
    
    # Debug the mesh
    mesh.compute_triangle_normals()
    normals = np.asarray(mesh.triangle_normals)
    horizontal_faces = np.sum(np.abs(normals[:, 2]) > 0.9)
    
    print(f"Ultra-clean mesh debug:")
    print(f"  Triangles: {len(mesh.triangles)}")
    print(f"  Horizontal faces: {horizontal_faces}")
    print(f"  Expected: 2 (ground only)")
    
    # Further clean if needed
    if horizontal_faces > 2:
        print("Removing excess horizontal faces...")
        mesh = completely_remove_horizontal_faces(mesh, z_threshold=0.8)
        
        # Re-add just the ground
        ground_vertices = np.array([
            [-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0]
        ])
        ground_triangles = np.array([[0, 1, 2], [0, 2, 3]])
        
        ground_mesh = o3d.geometry.TriangleMesh()
        ground_mesh.vertices = o3d.utility.Vector3dVector(ground_vertices)
        ground_mesh.triangles = o3d.utility.Vector3iVector(ground_triangles)
        ground_mesh.paint_uniform_color([0.6, 0.4, 0.2])
        
        mesh = mesh + ground_mesh
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()
    
    # Save and test
    mesh_filename = "ultra_clean.obj"
    o3d.io.write_triangle_mesh(mesh_filename, mesh, write_ascii=True)
    
    bounds = mesh.get_axis_aligned_bounding_box()
    agent_start = bounds.max_bound[2] + 2
    
    mjcf = f"""
<mujoco model="ultra_clean">
    <compiler angle="degree" meshdir="." />
    <option timestep="0.01" gravity="0 0 -9.81"/>
    
    <asset>
        <mesh name="clean_mesh" file="{mesh_filename}"/>
    </asset>
    
    <worldbody>
        <geom name="environment" type="mesh" mesh="clean_mesh" pos="0 0 0" 
              rgba="0.7 0.7 0.7 1" 
              contype="1" conaffinity="1"/>
        
        <body name="agent" pos="0 0 {agent_start}">
            <freejoint/>
            <geom name="agent_body" type="sphere" size="0.2" 
                  rgba="1 0 0 1"
                  contype="2" conaffinity="1"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.08 0.08 0.08"/>
        </body>
    </worldbody>
    
    <contact>
        <pair geom1="agent_body" geom2="environment"/>
    </contact>
</mujoco>
"""
    
    try:
        model = mujoco.MjModel.from_xml_string(mjcf)
        data = mujoco.MjData(model)
        
        print("✅ Ultra-clean mesh loaded!")
        
        for step in range(500):
            mujoco.mj_step(model, data)
            if step % 100 == 0:
                print(f"t={step/100:.0f}s: Z={data.qpos[2]:.3f}")
        
        return data.qpos[2] < 1.0  # Should be near ground
        
    except Exception as e:
        print(f"❌ Ultra-clean failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing multiple approaches for your MuJoCo version...\n")
    
    # Test 1: Primitive collision (should definitely work)
    success1 = test_primitive_collision_alternative()
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Ultra-clean mesh
    success2 = test_ultra_clean_mesh()
    
    print(f"\nResults:")
    print(f"Primitive collision: {'✅ SUCCESS' if success1 else '❌ FAILED'}")
    print(f"Ultra-clean mesh: {'✅ SUCCESS' if success2 else '❌ FAILED'}")
    
    if success1:
        print("\n✅ Use primitive collision for your real environment!")
    elif success2:
        print("\n✅ Ultra-clean mesh works - apply same cleaning to your point cloud!")
    else:
        print("\n❌ Both failed - there may be a deeper MuJoCo configuration issue.")