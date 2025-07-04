from .image_registration import DeepClipFeatureMatchingCOLMAP
from lightglue import DISK, SuperPoint, LightGlue
import torch 
import cv2 
import numpy as np
import pycolmap
from functools import partial
from pathlib import Path 
import shutil

def reconstruct_video(video_path, output_path=None):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap.release()
	duration = int(frame_count / fps)

	extractor = DISK(max_num_keypoints=2048).eval().to(device)
	matcher = LightGlue(features='disk').eval().to(device)

	matching_options = pycolmap.SequentialMatchingOptions()
	pairgenerator = partial(pycolmap.SequentialPairGenerator, options=matching_options)

	registration = DeepClipFeatureMatchingCOLMAP(
		vid_path=output_path if output_path else "deepsfm_output",
		clips=[[0, duration]],
		feature_extractor=extractor,
		feature_matcher=matcher,
		device=device,
		pairgenerator=pairgenerator,
		camera_model="SIMPLE_PINHOLE",
	)

	reconstructions = registration.register()
	if output_path is not None:
		output_path = Path(output_path) / "sparse"
		output_path.mkdir(exist_ok=True, parents=True)
		if reconstructions:
			for idx in reconstructions:
				r_dir = output_path / str(idx)
				r_dir.mkdir(exist_ok=True, parents=True)
				reconstructions[idx].write(str(r_dir))
							
	return reconstructions


def reconstruct_images(images_path, output_path=None, one_camera=True, device=None, **registration_class_kwargs):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
	extractor = DISK(max_num_keypoints=2048).eval().to(device)
	matcher = LightGlue(features='disk').eval().to(device)

	matching_options = pycolmap.ExhaustiveMatchingOptions()
	pairgenerator = partial(pycolmap.ExhaustivePairGenerator, options=matching_options)

	if one_camera:
		camera_mode = pycolmap.CameraMode.SINGLE
	else:
		camera_mode = pycolmap.CameraMode.AUTO

	if output_path is None:
		output_path = images_path.parent / "reconstruction"
		output_path.mkdir(exist_ok=True, parents=True)
		recon_path = output_path / "sparse"
		if recon_path.exists():
			print(f"Output path {recon_path} already exists, trying to read existing reconstruction...")
			recons = {}

			for recon_dir in recon_path.iterdir():
				# make sure points3D.bin, images.bin, and cameras.bin exist
				if not all([(recon_dir / "points3D.bin").exists(), (recon_dir / "images.bin").exists(), (recon_dir / "cameras.bin").exists()]):
					print(f"Failed to read existing reconstruction, starting new reconstruction...")
					shutil.rmtree(output_path)
					break

				recon = pycolmap.Reconstruction()
				try:
					recon.read(str(recon_dir))
					print(f"Successfully read existing reconstruction from {recon_path}")
					recons[int(recon_dir.name)] = recon
				except:
					print(f"Failed to read existing reconstruction, starting new reconstruction...")
					if recon_dir.exists():
						shutil.rmtree(output_path)

			if recons:
				return recons
	
	output_path.mkdir(exist_ok=True, parents=True)
	registration = DeepClipFeatureMatchingCOLMAP(
		images_path=images_path,
		feature_extractor=extractor,
		feature_matcher=matcher,
		device=device,
		pairgenerator=pairgenerator,
		camera_model="SIMPLE_PINHOLE",
		output_path=output_path,
		camera_mode=camera_mode,
		**registration_class_kwargs,
	)

	reconstructions = registration.register()
	if output_path is not None:
		output_path = Path(output_path) / "sparse"
		output_path.mkdir(exist_ok=True, parents=True)
		if reconstructions:
			for idx in reconstructions:
				r_dir = output_path / str(idx)
				r_dir.mkdir(exist_ok=True, parents=True)
				reconstructions[idx].write(str(r_dir))
							
	return reconstructions