import pycolmap
from deepsfm import reconstruct_images
from visualize3d import plot_reconstruction, init_figure
from pathlib import Path

path = Path("/Users/julian/Downloads/apple/12_90_489/images")

recon = reconstruct_images(path, output_path=path.parent / "reconstruction")
fig = init_figure()
plot_reconstruction(fig, recon, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
fig.show()