import pycolmap
import open3d as o3d 
import numpy as np

class CameraViewGroundDetector:
    def __init__(self, recon_path):
        """Detect ground using camera view geometry - bottom of image = ground"""
        self.reconstruction = pycolmap.Reconstruction(recon_path)
        self.points, self.colors = self._extract_points()
        
    def _extract_points(self):
        """Extract 3D points and colors"""
        points = []
        colors = []
        for point3d in self.reconstruction.points3D.values():
            points.append(point3d.xyz)
            colors.append(point3d.color / 255.0)
        return np.array(points), np.array(colors)
    
    def get_ground_points_from_image_bottom(self, bottom_fraction=0.3):
        """Find 3D points that project to the bottom portion of images"""
        
        ground_points = []
        ground_point_ids = set()
        
        print(f"Analyzing {len(self.reconstruction.images)} images...")
        print(f"Looking for points in bottom {bottom_fraction*100:.0f}% of images")
        
        for image_id, image in self.reconstruction.images.items():
            if not image.has_pose:
                continue
                
            # Get camera
            camera = self.reconstruction.cameras[image.camera_id]
            image_height = camera.height
            
            # Define bottom region of image (higher Y values = bottom in image coordinates)
            bottom_threshold = image_height * (1.0 - bottom_fraction)
            
            # Check each 2D point in this image
            points_in_bottom = 0
            for point2d in image.points2D:
                if point2d.has_point3D():
                    # Check if this 2D point is in bottom of image
                    y_coord = point2d.xy[1]
                    
                    if y_coord > bottom_threshold:  # Bottom part of image
                        point3d_id = point2d.point3D_id
                        
                        if point3d_id not in ground_point_ids:
                            ground_point_ids.add(point3d_id)
                            point3d = self.reconstruction.points3D[point3d_id]
                            ground_points.append(point3d.xyz)
                            points_in_bottom += 1
            
            if points_in_bottom > 0:
                print(f"  Image {image.name}: {points_in_bottom} bottom points")
        
        ground_points = np.array(ground_points)
        print(f"\nFound {len(ground_points)} potential ground points from image bottoms")
        
        return ground_points, ground_point_ids
    
    def filter_ground_points_by_height(self, ground_points, height_percentile=20):
        """Further filter ground points by keeping only the lowest ones"""
        
        if len(ground_points) == 0:
            return ground_points
        
        # Keep only points in bottom height percentile
        height_threshold = np.percentile(ground_points[:, 2], height_percentile)
        low_points = ground_points[ground_points[:, 2] <= height_threshold]
        
        print(f"Height filtering: {len(low_points)} / {len(ground_points)} points below {height_threshold:.2f}")
        
        return low_points
    
    def fit_ground_plane_to_points(self, ground_points):
        """Fit plane to ground points using least squares"""
        
        if len(ground_points) < 3:
            print("Not enough ground points to fit plane")
            return None, None
        
        # Center the points
        centroid = np.mean(ground_points, axis=0)
        centered_points = ground_points - centroid
        
        # SVD to find best-fit plane
        _, _, V = np.linalg.svd(centered_points)
        normal = V[-1]  # Normal is last row
        
        # Ensure normal points up
        if normal[2] < 0:
            normal = -normal
        
        print(f"Ground plane fit:")
        print(f"  Centroid: {centroid}")
        print(f"  Normal: {normal}")
        print(f"  Points used: {len(ground_points)}")
        
        # Calculate fit quality
        distances = np.abs(np.dot(centered_points, normal))
        mean_error = np.mean(distances)
        print(f"  Mean distance to plane: {mean_error:.3f}")
        
        return normal, centroid
    
    def create_oriented_ground_mesh(self, plane_normal, plane_point, size=20, resolution=0.5):
        """Create ground mesh aligned with the detected plane"""
        
        if abs(plane_normal[2]) > 0.9:  # Nearly vertical normal
            reference = np.array([1, 0, 0])
        else:
            reference = np.array([0, 0, 1])
        
        # Create orthogonal basis
        x_axis = np.cross(plane_normal, reference)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(plane_normal, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Create grid points in plane coordinate system
        half_size = size / 2
        coords = np.arange(-half_size, half_size, resolution)
        
        vertices = []
        faces = []
        vertex_idx = 0
        
        for i in range(len(coords) - 1):
            for j in range(len(coords) - 1):
                # Create quad in plane coordinates
                u1, u2 = coords[i], coords[i+1]
                v1, v2 = coords[j], coords[j+1]
                
                # Convert to world coordinates
                def plane_to_world(u, v):
                    return plane_point + u * x_axis + v * y_axis
                
                v1_world = plane_to_world(u1, v1)
                v2_world = plane_to_world(u2, v1)
                v3_world = plane_to_world(u2, v2)
                v4_world = plane_to_world(u1, v2)
                
                vertices.extend([v1_world, v2_world, v3_world, v4_world])
                
                # Two triangles
                faces.extend([
                    [vertex_idx, vertex_idx+1, vertex_idx+2],
                    [vertex_idx, vertex_idx+2, vertex_idx+3]
                ])
                vertex_idx += 4
        
        # Create mesh
        ground_mesh = o3d.geometry.TriangleMesh()
        ground_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        ground_mesh.triangles = o3d.utility.Vector3iVector(faces)
        ground_mesh.paint_uniform_color([0.8, 0.6, 0.4])  # Brown
        ground_mesh.compute_vertex_normals()
        
        return ground_mesh
    
    def normalize_reconstruction_to_ground(self, plane_normal, plane_point, target_size=10.0):
        """Normalize the reconstruction so ground is at minimum Z and properly oriented"""
        
        print("=== Normalizing Reconstruction to Ground Plane ===")
        target_up = np.array([0, 0, 1])
        
        if np.allclose(plane_normal, target_up):
            rotation_matrix = np.eye(3)
        elif np.allclose(plane_normal, -target_up):
            rotation_matrix = -np.eye(3)
        else:
            v = np.cross(plane_normal, target_up)
            s = np.linalg.norm(v)
            c = np.dot(plane_normal, target_up)
            
            vx = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]], 
                          [-v[1], v[0], 0]])
            
            rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
        
        # Step 2: Apply rotation to everything
        rotated_points = (rotation_matrix @ self.points.T).T
        rotated_plane_point = rotation_matrix @ plane_point
        
        # Step 3: Find minimum Z value after rotation to set ground there
        min_z = rotated_points[:, 2].min()
        translation = np.array([0, 0, -min_z])
        
        # Step 4: Center XY around origin
        xy_center = rotated_points[:, :2].mean(axis=0)
        translation[:2] = -xy_center
        
        # Apply translation
        normalized_points = rotated_points + translation
        
        # Step 5: Scale to target size
        scene_size = np.max(normalized_points.max(axis=0) - normalized_points.min(axis=0))
        scale_factor = target_size / scene_size
        normalized_points *= scale_factor
        
        # Transform camera positions too
        camera_positions = []
        camera_directions = []
        for image in self.reconstruction.images.values():
            if image.has_pose:
                # Position
                pos = image.projection_center()
                rotated_pos = rotation_matrix @ pos + translation
                normalized_pos = rotated_pos * scale_factor
                camera_positions.append(normalized_pos)
                
                # Direction
                direction = image.viewing_direction()
                rotated_dir = rotation_matrix @ direction
                camera_directions.append(rotated_dir)
        
        return {
            'points': normalized_points,
            'colors': self.colors,
            'camera_positions': np.array(camera_positions),
            'camera_directions': np.array(camera_directions),
            'transform': {
                'rotation': rotation_matrix,
                'translation': translation,
                'scale': scale_factor
            }
        }
    
    def detect_ground_and_normalize(self):
        """Complete pipeline: detect ground from camera views and normalize"""
        
        ground_points, ground_point_ids = self.get_ground_points_from_image_bottom(bottom_fraction=0.3)
        
        if len(ground_points) < 10:
            print("Too few ground points detected from image bottoms")
            return None
        
        filtered_ground_points = self.filter_ground_points_by_height(ground_points, height_percentile=30)
        
        if len(filtered_ground_points) < 10:
            print("Too few ground points after height filtering")
            return None
        
        plane_normal, plane_point = self.fit_ground_plane_to_points(filtered_ground_points)
        
        if plane_normal is None:
            print("Failed to fit ground plane")
            return None
        
        # Step 4: Create ground mesh
        ground_mesh = self.create_oriented_ground_mesh(plane_normal, plane_point, size=15)
        
        # Step 5: Normalize reconstruction
        normalized_data = self.normalize_reconstruction_to_ground(plane_normal, plane_point)
        
        return {
            'ground_points': filtered_ground_points,
            'ground_point_ids': ground_point_ids,
            'plane_normal': plane_normal,
            'plane_point': plane_point,
            'ground_mesh': ground_mesh,
            'normalized_data': normalized_data
        }
    
    def visualize_results(self, results):
        """Visualize the ground detection and normalization results"""
        
        if results is None:
            print("No results to visualize")
            return
        
        # Original reconstruction
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(self.points)
        original_pcd.colors = o3d.utility.Vector3dVector(self.colors)
        
        # Ground points
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(results['ground_points'])
        ground_pcd.paint_uniform_color([1, 0, 0])  # Red for ground
        
        print("=== Original Reconstruction ===")
        o3d.visualization.draw_geometries(
            [original_pcd, ground_pcd],
            window_name="Original: Red = Ground Points from Image Bottoms",
            width=800, height=600
        )
        
        # Normalized reconstruction
        normalized_pcd = o3d.geometry.PointCloud()
        normalized_pcd.points = o3d.utility.Vector3dVector(results['normalized_data']['points'])
        normalized_pcd.colors = o3d.utility.Vector3dVector(results['normalized_data']['colors'])
        
        # Camera positions in normalized space
        camera_spheres = []
        for pos in results['normalized_data']['camera_positions']:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.translate(pos)
            sphere.paint_uniform_color([0, 1, 0])  # Green cameras
            camera_spheres.append(sphere)
        
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        
        print("=== Normalized Reconstruction ===")
        print("Green spheres = cameras, Ground at minimum Z")
        
        # Only show point cloud, cameras and coordinate frame - no artificial ground plane
        geometries = [normalized_pcd, coordinate_frame] + camera_spheres
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Normalized: Ground at minimum Z, Cameras in Green",
            width=1200, height=800
        )

class RLMeshCreator:
    """Creates separate mesh components for MuJoCo RL environments from SfM data"""
    
    def __init__(self, normalized_data: dict[str, np.ndarray]):
        self.points = normalized_data['points']
        self.colors = normalized_data['colors']
        self.camera_positions = normalized_data.get('camera_positions', [])
        self.min_z = self.points[:, 2].min()
        
    def create_separate_meshes(self, add_walls=True, simplify_environment=True):
        """
        Create separate mesh components for proper MuJoCo collision.
        
        Returns:
            dict: Dictionary of meshes with keys like 'ground', 'wall_0', 'environment'
        """
        meshes = {}
        
        # 1. Ground plane
        meshes['ground'] = self.create_thick_ground_plane()
        
        # 2. Boundary walls
        if add_walls:
            walls = self.create_boundary_walls()
            for i, wall in enumerate(walls):
                meshes[f'wall_{i}'] = wall
        
        # 3. Environment mesh from point cloud
        env_mesh = self.create_environment_mesh(simplify=simplify_environment)
        if env_mesh is not None:
            meshes['environment'] = env_mesh
        
        return meshes
    
    def create_thick_ground_plane(self, size=20, thickness=0.3):
        """Create a thick ground plane mesh"""
        ground = o3d.geometry.TriangleMesh.create_box(size, size, thickness)
        # Place ground so its TOP surface is at min_z
        ground.translate([-size/2, -size/2, self.min_z - thickness])
        ground.paint_uniform_color([0.4, 0.3, 0.2])  # Dark brown
        ground.compute_vertex_normals()
        print(f"Created thick ground: top at Z={self.min_z:.3f}, bottom at Z={self.min_z - thickness:.3f}")
        return ground
    
    def create_boundary_walls(self, height=4.0, thickness=0.3):
        """Create boundary wall meshes"""
        bounds_min = self.points.min(axis=0)
        bounds_max = self.points.max(axis=0)
        
        # Extend bounds slightly
        margin = 1.0
        x_min, y_min = bounds_min[:2] - margin
        x_max, y_max = bounds_max[:2] + margin
        
        walls = []
        
        # 4 boundary walls
        wall_configs = [
            # position, size
            ([x_min - thickness, y_min, self.min_z], [thickness, y_max - y_min, height]),  # Left
            ([x_max, y_min, self.min_z], [thickness, y_max - y_min, height]),              # Right  
            ([x_min, y_min - thickness, self.min_z], [x_max - x_min, thickness, height]),  # Front
            ([x_min, y_max, self.min_z], [x_max - x_min, thickness, height])               # Back
        ]
        
        for pos, size in wall_configs:
            wall = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2])
            wall.translate(pos)
            wall.paint_uniform_color([0.8, 0.8, 0.9])
            wall.compute_vertex_normals()
            walls.append(wall)
        
        print(f"Created {len(walls)} boundary walls, height={height}")
        return walls
    
    def create_environment_mesh(self, simplify=True):
        """Create environment mesh from point cloud (building geometry)"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        
        mesh = None
        
        # Try alpha shapes with different parameters
        for alpha in [0.3, 0.5, 0.8]:
            try:
                test_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
                if len(test_mesh.triangles) > 50:
                    mesh = test_mesh
                    print(f"Created mesh with alpha={alpha}, triangles={len(mesh.triangles)}")
                    break
            except:
                continue
        
        if mesh is None:
            print("Failed to create environment mesh from point cloud")
            return None
        
        # Process the mesh
        if simplify:
            mesh = self.simplify_mesh(mesh)
        
        # Remove horizontal faces to prevent ceiling effects
        mesh = self.remove_horizontal_faces(mesh)
        mesh = self.remove_top_faces(mesh)
        
        # Final cleanup
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()
        
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.7, 0.7, 0.7])
        
        print(f"Final environment mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        return mesh
    
    def simplify_mesh(self, mesh, target_triangles=2000):
        """Simplify mesh for better collision performance"""
        # Remove small disconnected components
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        # Keep only large clusters
        large_cluster_ids = np.where(cluster_n_triangles > 20)[0]
        triangles_to_keep = np.isin(triangle_clusters, large_cluster_ids)
        
        mesh.remove_triangles_by_mask(~triangles_to_keep)
        mesh.remove_unreferenced_vertices()
        
        # Simplify
        if len(mesh.triangles) > target_triangles:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
        
        print(f"Simplified mesh to {len(mesh.triangles)} triangles")
        return mesh
    
    def remove_horizontal_faces(self, mesh, normal_threshold=0.7):
        """Remove faces with normals pointing up (horizontal surfaces)"""
        mesh.compute_triangle_normals()
        triangle_normals = np.asarray(mesh.triangle_normals)
        triangles = np.asarray(mesh.triangles)
        
        # Keep triangles where normal Z component is less than threshold
        keep_mask = triangle_normals[:, 2] < normal_threshold
        filtered_triangles = triangles[keep_mask]
        
        removed_count = len(triangles) - len(filtered_triangles)
        if removed_count > 0:
            print(f"Removed {removed_count} horizontal faces")
            
            new_mesh = o3d.geometry.TriangleMesh()
            new_mesh.vertices = mesh.vertices
            new_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)
            if mesh.has_vertex_colors():
                new_mesh.vertex_colors = mesh.vertex_colors
            return new_mesh
        
        return mesh
    
    def remove_top_faces(self, mesh, z_threshold_percentile=80):
        """Remove faces that are too high (potential ceilings)"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Calculate Z threshold
        z_threshold = np.percentile(vertices[:, 2], z_threshold_percentile)
        
        # Get triangle centers
        triangle_centers = np.array([vertices[tri].mean(axis=0) for tri in triangles])
        
        # Keep only triangles below threshold
        keep_mask = triangle_centers[:, 2] < z_threshold
        filtered_triangles = triangles[keep_mask]
        
        removed_count = len(triangles) - len(filtered_triangles)
        if removed_count > 0:
            print(f"Removed {removed_count} top faces above Z={z_threshold:.2f}")
            
            new_mesh = o3d.geometry.TriangleMesh()
            new_mesh.vertices = mesh.vertices
            new_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)
            if mesh.has_vertex_colors():
                new_mesh.vertex_colors = mesh.vertex_colors
            return new_mesh
        
        return mesh
    
    def create_camera_markers(self, radius=0.1):
        """Create sphere markers for camera positions"""
        camera_spheres = []
        for i, pos in enumerate(self.camera_positions):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.translate(pos)
            sphere.paint_uniform_color([1, 0, 0])  # Red
            camera_spheres.append(sphere)
        return camera_spheres
    
    def visualize_all_meshes(self, meshes):
        """Visualize all mesh components in Open3D"""
        geometries = list(meshes.values())
        
        # Add camera markers
        geometries.extend(self.create_camera_markers(radius=0.05))
        
        # Add coordinate frame
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
        
        print("\nOpen3D Visualization:")
        print("- Brown: Ground plane")
        print("- Light blue: Boundary walls")
        print("- Gray: Environment mesh (building)")
        print("- Red: Camera positions")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name="RL Environment Components",
            width=1200, height=800
        )

def create_rl_environment(normalized_data, add_walls=True, simplify=True, visualize=True):
    """
    Create RL environment with separate mesh components.
    
    Args:
        normalized_data: Dictionary with 'points', 'colors', 'camera_positions'
        add_walls: Whether to add boundary walls
        simplify: Whether to simplify the environment mesh
        visualize: Whether to show Open3D visualization
        
    Returns:
        dict: Contains 'meshes' dict and 'creator' object
    """
    creator = RLMeshCreator(normalized_data)
    
    print("Creating RL environment with separate meshes...")
    meshes = creator.create_separate_meshes(add_walls=add_walls, simplify_environment=simplify)
    
    if visualize:
        creator.visualize_all_meshes(meshes)
    
    return {
        'meshes': meshes,
        'creator': creator
    }