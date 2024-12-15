import os
import subprocess

def run_colmap_dense_reconstruction(project_path, dense_workspace_dir, output_dense_dir):
    """
    Runs dense reconstruction using COLMAP CLI.

    Parameters:
        project_path (str): Path to the COLMAP sparse reconstruction project folder.
        dense_workspace_dir (str): Directory containing images and sparse reconstruction.
        output_dense_dir (str): Directory to store the dense reconstruction results.
    """

    # Step 1: Create the dense workspace directory if it does not exist
    if not os.path.exists(output_dense_dir):
        os.makedirs(output_dense_dir)
        print(f"Created output directory: {output_dense_dir}")

    # Step 2: Run the dense stereo reconstruction
    dense_stereo_command = [
        "colmap", "stereo",
        "--workspace_path", dense_workspace_dir,
        "--workspace_format", "COLMAP",
        "--DenseStereo.geom_consistency", "true"
    ]
    print("Running Dense Stereo Reconstruction...")
    subprocess.run(dense_stereo_command, check=True)
    print("Dense stereo reconstruction completed.")

    # Step 3: Run the dense fusion
    dense_fusion_command = [
        "colmap", "stereo_fusion",
        "--workspace_path", dense_workspace_dir,
        "--workspace_format", "COLMAP",
        "--input_type", "geometric",
        "--output_path", os.path.join(output_dense_dir, "fused.ply")
    ]
    print("Running Dense Fusion...")
    subprocess.run(dense_fusion_command, check=True)
    print("Dense fusion completed.")

    # Step 4: Optionally, run Poisson meshing to create a mesh from the point cloud
    poisson_mesh_command = [
        "colmap", "poisson_meshing",
        "--input_path", os.path.join(output_dense_dir, "fused.ply"),
        "--output_path", os.path.join(output_dense_dir, "meshed-poisson.ply")
    ]
    print("Running Poisson Meshing...")
    subprocess.run(poisson_mesh_command, check=True)
    print("Poisson meshing completed.")

if __name__ == "__main__":
    # Specify paths
    sparse_project_dir = "/path/to/sparse/project"  # Update with your sparse reconstruction path
    dense_workspace = "/path/to/dense/workspace"   # Directory with images and sparse reconstruction
    dense_output_dir = "/path/to/dense/output"     # Where dense outputs will be stored

    # Run the dense reconstruction
    run_colmap_dense_reconstruction(sparse_project_dir, dense_workspace, dense_output_dir)
