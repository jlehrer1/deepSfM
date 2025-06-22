import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os 

device = "cpu"
dtype = torch.bfloat16 if device == "cuda:0" else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
img_dir = "images"
print("Loading images...")
image_names = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if ".DS" not in f]
images = load_and_preprocess_images(image_names).to(device)

print("Loaded images...")
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
        breakpoint()