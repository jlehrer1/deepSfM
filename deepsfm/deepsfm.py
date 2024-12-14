from image_registration import DeepClipFeatureMatchingCOLMAP
from lightglue import DISK, SuperPoint, LightGlue
import torch 
import cv2 
import numpy as np
import pycolmap
from functools import partial
from pathlib import Path 

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
		output_path = Path(output_path)
		output_path.mkdir(exist_ok=True, parents=True)

		if reconstructions:
			for idx in reconstructions:
				r_dir = output_path / f"reconstruction_{idx}"
				r_dir.mkdir(exist_ok=True, parents=True)
				reconstructions[idx].write(str(r_dir))
							
	return reconstructions


def reconstruct_images(images_path, output_path=None, one_camera=False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	extractor = DISK(max_num_keypoints=2048).eval().to(device)
	matcher = LightGlue(features='disk').eval().to(device)

	matching_options = pycolmap.ExhaustiveMatchingOptions()
	pairgenerator = partial(pycolmap.ExhaustivePairGenerator, options=matching_options)

	if one_camera:
		camera_mode = pycolmap.CameraMode.SINGLE
	else:
		camera_mode = pycolmap.CameraMode.AUTO

	registration = DeepClipFeatureMatchingCOLMAP(
		images_path=images_path,
		feature_extractor=extractor,
		feature_matcher=matcher,
		device=device,
		pairgenerator=pairgenerator,
		camera_model="SIMPLE_PINHOLE",
		output_path=output_path if output_path else "deepsfm_output",
		camera_mode=camera_mode,
	)

	reconstructions = registration.register()
	if output_path is not None:
		output_path = Path(output_path)
		output_path.mkdir(exist_ok=True, parents=True)
		if reconstructions:
			for idx in reconstructions:
				r_dir = output_path / f"reconstruction_{idx}"
				r_dir.mkdir(exist_ok=True, parents=True)
				reconstructions[idx].write(str(r_dir))
							
	return reconstructions