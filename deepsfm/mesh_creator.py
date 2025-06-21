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
from sklearn.cluster import DBSCAN

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
        
        # Create a local coordinate system for the plane
        # Normal is Z-axis, need to find X and Y axes
        
        # Find a vector that's not parallel to normal
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
        
        # Step 1: Rotation to align ground normal with Z-axis
        target_up = np.array([0, 0, 1])
        
        if np.allclose(plane_normal, target_up):
            rotation_matrix = np.eye(3)
        elif np.allclose(plane_normal, -target_up):
            rotation_matrix = -np.eye(3)
        else:
            # Rodrigues rotation formula
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
        
        print(f"Normalization complete:")
        print(f"  Rotation applied to align ground normal")
        print(f"  Translation: {translation}")
        print(f"  Scale factor: {scale_factor:.3f}")
        print(f"  Final bounds: {normalized_points.min(axis=0)} to {normalized_points.max(axis=0)}")
        print(f"  Ground placed at Z = {normalized_points[:, 2].min():.3f}")
        
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
        
        print("=== Ground Detection Using Camera View Geometry ===")
        
        # Step 1: Get points from bottom of images
        ground_points, ground_point_ids = self.get_ground_points_from_image_bottom(bottom_fraction=0.3)
        
        if len(ground_points) < 10:
            print("Too few ground points detected from image bottoms")
            return None
        
        # Step 2: Filter by height
        filtered_ground_points = self.filter_ground_points_by_height(ground_points, height_percentile=30)
        
        if len(filtered_ground_points) < 10:
            print("Too few ground points after height filtering")
            return None
        
        # Step 3: Fit plane to ground points
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
    def __init__(self, normalized_data):
        self.points = normalized_data['points']
        self.colors = normalized_data['colors']
        self.camera_positions = normalized_data.get('camera_positions', [])
        self.min_z = self.points[:, 2].min()
        
    def remove_top_faces_from_mesh(self, mesh, z_threshold_percentile=90):
        """Remove triangles that form tops/ceilings"""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Calculate Z threshold - remove triangles above this
        z_threshold = np.percentile(vertices[:, 2], z_threshold_percentile)
        
        # Get triangle centers
        triangle_centers = []
        for tri in triangles:
            center = vertices[tri].mean(axis=0)
            triangle_centers.append(center)
        triangle_centers = np.array(triangle_centers)
        
        # Keep only triangles below threshold
        keep_mask = triangle_centers[:, 2] < z_threshold
        filtered_triangles = triangles[keep_mask]
        
        print(f"Removed {len(triangles) - len(filtered_triangles)} top faces above Z={z_threshold:.2f}")
        
        # Create new mesh
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = mesh.vertices
        new_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)
        new_mesh.vertex_colors = mesh.vertex_colors
        
        return new_mesh
    
    def remove_horizontal_faces_by_normal(self, mesh, normal_threshold=0.7):
        """Remove triangles with normals pointing up (horizontal faces)"""
        mesh.compute_triangle_normals()
        triangle_normals = np.asarray(mesh.triangle_normals)
        triangles = np.asarray(mesh.triangles)
        
        # Keep triangles where normal Z component is less than threshold
        # (i.e., not pointing straight up)
        keep_mask = triangle_normals[:, 2] < normal_threshold
        filtered_triangles = triangles[keep_mask]
        
        print(f"Removed {len(triangles) - len(filtered_triangles)} horizontal faces by normal")
        
        new_mesh = o3d.geometry.TriangleMesh()
        new_mesh.vertices = mesh.vertices
        new_mesh.triangles = o3d.utility.Vector3iVector(filtered_triangles)
        new_mesh.vertex_colors = mesh.vertex_colors
        
        return new_mesh
    
    def create_environment_mesh_no_roof(self, add_walls=True):
        """Create environment mesh WITHOUT any roof/ceiling"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

        mesh = None
        
        # Try alpha shapes first (but remove tops)
        for alpha in [0.2, 0.3, 0.5, 0.8]:
            try:
                test_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
                if len(test_mesh.triangles) > 50:
                    # Remove top faces from alpha shape
                    mesh = self.remove_horizontal_faces_by_normal(test_mesh)
                    mesh = self.remove_top_faces_from_mesh(mesh)
                    print(f"Alpha shapes: alpha={alpha}, triangles after roof removal={len(mesh.triangles)}")
                    break
            except:
                continue

        # Fallback: Ball pivoting (but remove tops)
        if mesh is None or len(mesh.triangles) < 20:
            try:
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                radii = [avg_dist * 0.5, avg_dist, avg_dist * 2]
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii)
                )
                mesh = self.remove_horizontal_faces_by_normal(mesh)
                mesh = self.remove_top_faces_from_mesh(mesh)
                print(f"Ball pivoting: triangles after roof removal={len(mesh.triangles)}")
            except:
                pass

        # Last resort: Create walls only (NO full box)
        if mesh is None or len(mesh.triangles) == 0:
            mesh = self._create_walls_only_mesh()

        # Create proper thick ground plane
        ground_mesh = self.create_thick_ground_plane()

        # Add boundary walls if requested
        walls = []
        if add_walls:
            walls = self.create_boundary_walls()

        # Combine all (environment + ground + walls)
        combined_mesh = mesh + ground_mesh
        for wall in walls:
            combined_mesh += wall

        # Cleanup
        combined_mesh.remove_degenerate_triangles()
        combined_mesh.remove_duplicated_triangles()
        combined_mesh.remove_duplicated_vertices()
        combined_mesh.compute_vertex_normals()

        if not combined_mesh.has_vertex_colors():
            combined_mesh.paint_uniform_color([0.7, 0.7, 0.7])

        return combined_mesh

    def _create_walls_only_mesh(self):
        """Create walls from point cloud clusters without tops"""
        clustering = DBSCAN(eps=0.5, min_samples=10)
        labels = clustering.fit_predict(self.points)
        
        meshes = []
        for cluster_id in range(max(labels) + 1):
            cluster_mask = labels == cluster_id
            cluster_points = self.points[cluster_mask]
            
            if len(cluster_points) < 10:
                continue
                
            try:
                # Create a wall-like structure from cluster
                wall_mesh = self._create_vertical_wall_from_points(cluster_points)
                if wall_mesh is not None:
                    meshes.append(wall_mesh)
            except:
                continue
        
        if meshes:
            combined = meshes[0]
            for mesh in meshes[1:]:
                combined += mesh
            return combined
        else:
            # Return empty mesh rather than full box
            return o3d.geometry.TriangleMesh()
    
    def _create_vertical_wall_from_points(self, points):
        """Create vertical wall geometry from point cluster"""
        if len(points) < 4:
            return None
            
        # Project points to XY plane for 2D convex hull
        points_2d = points[:, :2]
        
        # Simple bounding rectangle approach
        min_x, min_y = points_2d.min(axis=0)
        max_x, max_y = points_2d.max(axis=0)
        
        # Get height range
        min_z = points[:, 2].min()
        max_z = points[:, 2].max()
        
        # Create vertical quads around the perimeter
        thickness = 0.1
        height = max_z - min_z
        
        if height < 0.2:  # Skip very flat clusters
            return None
        
        # Create 4 walls around the bounding rectangle
        walls = []
        
        # Front wall
        wall = o3d.geometry.TriangleMesh.create_box(max_x - min_x, thickness, height)
        wall.translate([min_x, min_y - thickness/2, min_z])
        walls.append(wall)
        
        # Back wall  
        wall = o3d.geometry.TriangleMesh.create_box(max_x - min_x, thickness, height)
        wall.translate([min_x, max_y - thickness/2, min_z])
        walls.append(wall)
        
        # Left wall
        wall = o3d.geometry.TriangleMesh.create_box(thickness, max_y - min_y, height)
        wall.translate([min_x - thickness/2, min_y, min_z])
        walls.append(wall)
        
        # Right wall
        wall = o3d.geometry.TriangleMesh.create_box(thickness, max_y - min_y, height)
        wall.translate([max_x - thickness/2, min_y, min_z])
        walls.append(wall)
        
        # Combine walls
        combined = walls[0]
        for wall in walls[1:]:
            combined += wall
            
        combined.paint_uniform_color([0.6, 0.6, 0.8])
        return combined
    
    def create_thick_ground_plane(self, size=20, thickness=0.5):
        """Create a THICK, solid ground plane"""
        ground = o3d.geometry.TriangleMesh.create_box(size, size, thickness)
        # Place ground so its TOP surface is at min_z
        ground.translate([-size/2, -size/2, self.min_z - thickness])
        ground.paint_uniform_color([0.4, 0.3, 0.2])  # Dark brown
        ground.compute_vertex_normals()
        print(f"Created thick ground: top at Z={self.min_z:.3f}, thickness={thickness}")
        return ground
    
    def create_boundary_walls(self, height=4.0, thickness=0.2):
        """Create boundary walls around the scene"""
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
            ([x_min, y_min - thickness, self.min_z], [x_max - x_min, thickness, height]), # Front
            ([x_min, y_max, self.min_z], [x_max - x_min, thickness, height])              # Back
        ]
        
        for pos, size in wall_configs:
            wall = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2])
            wall.translate(pos)
            wall.paint_uniform_color([0.8, 0.8, 0.9])
            wall.compute_vertex_normals()
            walls.append(wall)
        
        print(f"Created {len(walls)} boundary walls, height={height}")
        return walls
    
def create_fixed_rl_environment(normalized_data, add_walls=True, visualize=True):
    """Create RL environment with NO invisible roof"""
    creator = RLMeshCreator(normalized_data)
    
    print("Creating environment mesh WITHOUT roof...")
    env_mesh = creator.create_environment_mesh_no_roof(add_walls=add_walls)
    
    # Add camera visualization
    camera_spheres = []
    for pos in creator.camera_positions:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(pos)
        sphere.paint_uniform_color([1, 0, 0])
        camera_spheres.append(sphere)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    all_geometries = [env_mesh, coordinate_frame] + camera_spheres
    
    if visualize:
        print("\nVisualization:")
        print("- Gray: Environment mesh (NO ROOF)")
        print("- Brown: Thick ground plane") 
        print("- Light blue: Boundary walls")
        print("- Red spheres: Camera positions")
        
        o3d.visualization.draw_geometries(
            all_geometries,
            window_name="Fixed RL Environment - No Invisible Roof",
            width=1200, height=800
        )
    
    return {
        'environment_mesh': env_mesh,
        'camera_spheres': camera_spheres,
        'all_geometries': all_geometries
    }