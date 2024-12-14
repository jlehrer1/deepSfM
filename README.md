# deepSfM
Library for 3d reconstruction (structure from motion) from sets of 2d images or videos using deep learning based feature extraction + matching and triangulation + bundle adjustment via COLMAP.
The image matching is done with `LightGlue` which can either use `SuperPoint`, `DISK` or `ALIKED`. We use `DISK` for the example since it generally performs well and seems a bit less sensitive to outlier correspondences.

To reconstruct and visualize the pointcloud + camera positions from a set of scene images, run
```python
from deepsfm import reconstruct_images
from visualize3d import plot_reconstruction, init_figure

recon = reconstruct_images("../ex_imgs", single_camera=False)
fig = init_figure()
plot_reconstruction(fig, recon, color='rgba(255,0,0,0.5)', name="mapping", points_rgb=True)
fig.show()
```

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

