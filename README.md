# deepSfM-RL
Library for training RL models in Mujoco where we the environment is created from 3d reconstruction (structure from motion) from sets of 2d images or videos.

Uses deep learning based feature extraction + matching and triangulation + bundle adjustment via COLMAP, then creates a mesh using [Poisson surface reconstruction](https://hhoppe.com/poissonrecon.pdf)

The image matching is done with `LightGlue` which can either use `SuperPoint`, `DISK` or `ALIKED`. We use `DISK` for the example since it generally performs well and seems a bit less sensitive to outlier correspondences.

![DISK + LightGlue matching example](example.png)

To reconstruct and visualize the pointcloud + camera positions from a set of scene images, run
```python
from deepsfm import reconstruct_images
from visualize3d import plot_reconstruction, init_figure

img_path = "/path/to/images/"
recon = reconstruct_images(img_path, single_camera=False)
fig = init_figure()
plot_reconstruction(fig, recon, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
fig.show()
```

The reconstruction will automatically be saved in the parent direction of your images in the "reconstruction" subfolder, which can then be used for downstream tasks such as 3d Gaussian Splatting or NERF. 

To reconstruct and visualize a video, run
```python
from deepsfm import reconstruct_images
from visualize3d import plot_reconstruction, init_figure

recon = reconstruct_video("video.mp4")
fig = init_figure()
plot_reconstruction(fig, recon, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
fig.show()
```

These top-level functions are for ease of use, but the classes in image_registration.py provide more depth into the image matching and reconstruction process.

The plot reconstruction function will open a web page with an html rendering of your sparse pointcloud and camera frustums from the SfM process. 