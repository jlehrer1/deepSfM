import pycolmap
from deepsfm import reconstruct_images
from visualize3d import plot_reconstruction, init_figure
from pathlib import Path
from torchvision.transforms import Resize, Compose, PILToTensor


tform = Compose(
	[
		Resize((1000, 1500)),
		PILToTensor(),
	]
)

path = Path("/Users/julian/Downloads/room")

recon = reconstruct_images(path, output_path=path.parent / "reconstruction", one_camera=False)
fig = init_figure()
plot_reconstruction(fig, recon[0], color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
fig.show()