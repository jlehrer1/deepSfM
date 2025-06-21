import pycolmap
from deepsfm.deepsfm import reconstruct_images
from deepsfm.visualize3d import plot_reconstruction, init_figure
from pathlib import Path
from torchvision.transforms import Resize, Compose, PILToTensor

# path to images
path = Path("/teamspace/studios/this_studio/deepSfM/images")

recon = reconstruct_images(path, one_camera=False, device="cuda:0")

fig = init_figure()
plot_reconstruction(fig, recon[0], color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True, cameras=True)
fig.show()