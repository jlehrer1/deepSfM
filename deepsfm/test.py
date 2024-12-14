import pycolmap
from deepsfm import reconstruct_images
from visualize3d import plot_reconstruction, init_figure

recon = reconstruct_images("../ex_imgs")
fig = init_figure()
plot_reconstruction(fig, recon, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
fig.show()