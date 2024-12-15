import concurrent.futures
import os
import shutil
import tempfile
from functools import cached_property
from pathlib import Path
from typing import Any, Optional, Callable

import cv2
import numpy as np
import pycolmap
from tqdm import tqdm
import torch 
import torch.nn as nn
from lightglue.utils import load_image, rbd
from copy import deepcopy
from functools import partial

class ClipRegistration:
    VALID_IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]
    VALID_CAM_MODELS = ["SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "PINHOLE", "PINHOLE_FISHEYE"]
    def __init__(
        self,
        clips: list[tuple[int]] = None,
        vid_path: str = None,
        images_path: str = None,
        sample_rate: float = 1.0,
        transform: Optional[Callable] = None,
        output_path: Optional[str] = None,
    ):
        """
        Parameters:
        clips: list[tuple[int]]: List of tuples for clips in seconds, for example [[0, 5], [5, 10]] defines two
        clips from second 0 to 5 and then second 5 to 10.
        vid_path: str: Path to the video file on local disk
        sample_rate: float defining the proportion of frames to sample from each clip.
        transform: Tranform applied to images before saving to disk, for example edge detection and resizing
        """
        super().__init__()

        self.images_path = images_path
        self.clips = clips
        self.vid_path = vid_path

        assert vid_path and clips or images_path, "Either clips and vid_path or images_path must be provided, but not both."
        assert not (vid_path and images_path), "Either clips and vid_path or images_path must be provided, but not both."

        self.sample_rate = sample_rate
        self.transform = transform
        self.output_path = output_path

    def sample_clip(self, clip: tuple[int], output_dir: str):
        """
        Saves video frames from a clip defines as (start_sec, end_sec) into a given output directory

        Parameters:
        clip: tuple[int]: Clip defined from (start_time, end_time)
        output_dir: Output directory to save frames to
        """
        vid_path = str(self.vid_path)
        sample_rate = self.sample_rate

        start_sec, end_sec = clip
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = int(1 / sample_rate)
        frame_count = 0
        total_frames = int((end_sec - start_sec) * fps)

        start_frame = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_interval == 0:
                frame_num = start_frame + frame_count
                out_path = os.path.join(output_dir, f"{frame_num:08d}.jpg")
                if self.transform is not None:
                    frame = self.transform(frame)
                    assert isinstance(frame, np.ndarray), "Transform must return a numpy array"
                cv2.imwrite(out_path, frame)

            frame_count += 1

        cap.release()

        return output_dir

    def register_single_clip(self, *args, **kwargs):
        raise NotImplementedError

    def register(self, *args, **kwargs):
        raise NotImplementedError

    @cached_property
    def clip_midpoints(self):
        midpoints = np.array([(x[1] - x[0]) / 2 for x in self.clips])

        return midpoints


class COLMAPClipRegistration(ClipRegistration):
    """
    ClipRegistration subclass that uses COLMAP (https://colmap.github.io/) for 3d reconstruction
    given a set of 2d images and a camera model (structure-from-motion)
    """

    def __init__(
        self,
        clips: list[tuple[int]] = None,
        vid_path: str = None,
        images_path: str = None,
        sample_rate=1,
        transform: Optional[Callable] = None,
        sift_extraction_options: pycolmap.SiftExtractionOptions = pycolmap.SiftExtractionOptions(),
        sequential_matching_options: pycolmap.SequentialMatchingOptions = pycolmap.SequentialMatchingOptions(),
        mapping_options: pycolmap.IncrementalPipelineOptions = pycolmap.IncrementalPipelineOptions(),
        camera_model: str = "SIMPLE_RADIAL_FISHEYE",
        feature_extraction_kwargs: Optional[dict[str, Any]] = None,
        sequential_matching_kwargs: Optional[dict[str, Any]] = None,
        mapping_kwargs: Optional[dict[str, Any]] = None,
        num_threads: int = os.cpu_count(),
        output_path: str = "colmap_output",
    ):
        super().__init__(clips, vid_path, images_path, sample_rate, transform, output_path)
        self.image_registration_results: list[pycolmap.Reconstruction] = None
        self.sift_extraction_options = sift_extraction_options
        self.sequential_matching_options = sequential_matching_options
        self.mapping_options = mapping_options
        self.camera_model = camera_model
        self.feature_extraction_kwargs = feature_extraction_kwargs if feature_extraction_kwargs is not None else dict()
        self.sequential_matching_kwargs = (
            sequential_matching_kwargs if sequential_matching_kwargs is not None else dict()
        )
        self.mapping_kwargs = mapping_kwargs if mapping_kwargs is not None else dict()
        self.num_threads = num_threads

    def register_single_clip(
        self,
        clip: tuple[int],
        sift_extraction_options: pycolmap.SiftExtractionOptions,
        sequential_matching_options: pycolmap.SequentialMatchingOptions,
        mapping_options: pycolmap.IncrementalPipelineOptions,
        camera_model: str,
    ):
        output_path = Path(tempfile.mkdtemp())

        if not self.images_path:
            image_dir = output_path / "images"
            image_dir.mkdir(exist_ok=True, parents=True)
            print(f"Sampling clip {clip} from video...")
            self.sample_clip(clip, image_dir)
        else:
            image_dir = Path(self.images_path)

        database_path = output_path / "database.db"
        n_images = len(list(image_dir.glob("*")))  # potentially possible in future once we add filtering to sampling

        if n_images > 0:
            pycolmap.extract_features(  # extract features using SIFT
                database_path,
                image_dir,
                camera_model=camera_model,
                sift_options=sift_extraction_options,
                camera_mode=pycolmap.CameraMode.SINGLE,
                **self.feature_extraction_kwargs,
            )
            pycolmap.match_sequential(  # match features and keypoints between images
                database_path,
                matching_options=sequential_matching_options,
                **self.sequential_matching_kwargs,
            )
            maps = pycolmap.incremental_mapping(  # triangulation & bundle adjustment
                database_path,
                image_dir,
                output_path,
                options=mapping_options,
                **self.mapping_kwargs,
            )

            shutil.rmtree(output_path)

            return maps if len(maps) > 0 else None

        return None

    def register(
        self,
    ) -> dict[str, pycolmap.Reconstruction]:
        """
        Register all video clips in parallel using COLMAP.
        """

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for clip in self.clips:
                future = executor.submit(
                    self.register_single_clip,
                    clip,
                    self.sift_extraction_options,
                    self.sequential_matching_options,
                    self.mapping_options,
                    self.camera_model,
                )
                futures.append(future)

            results = []
            for future in tqdm(futures, desc="Registering clips"):
                results.append(future.result())

        self.image_registration_results = results
        return results

    @cached_property
    def camera_translations(self):
        """
        Returns a list of camera translations after registration
        """

        if self.image_registration_results is None:
            raise RuntimeError("Image registration has not been run. register() method must be called first.")

        cams = []
        for registration_result in self.image_registration_results:
            if registration_result is not None:
                translations = [list(img.cam_from_world.translation) for img in registration_result.images.values()]
                cams.append(translations)
            else:
                cams.append(None)
        return cams

    @cached_property
    def num_images(self) -> np.array:
        """
        Returns an array of the number of images succcessfully reconstructed for each clip
        """
        if self.image_registration_results is None:
            raise RuntimeError("Image registration has not been run. register() method must be called first.")

        imgs = []
        for registration_result in self.image_registration_results:
            imgs.append(len(registration_result.images) if registration_result else 0)

        return np.array(imgs)

    @cached_property
    def reconstructed_frames(self):
        """
        Returns the frame indices of the successfully reconstructed images in each clip.
        """
        if self.image_registration_results is None:
            raise RuntimeError("Image registration has not been run. register() method must be called first.")

        frames = []
        for registration_result in self.image_registration_results:
            curr_frames = []
            if registration_result is not None:
                curr_imgs = registration_result.images

                for img in curr_imgs:
                    frame = int(curr_imgs[img].name.split(".")[0])
                    curr_frames.append(frame)

            frames.append(curr_frames)

        return frames


class DeepClipFeatureMatchingCOLMAP(COLMAPClipRegistration):
    def __init__(
        self, 
        feature_extractor: nn.Module, 
        feature_matcher: nn.Module, 
        pairgenerator: partial[pycolmap.PairGenerator], 
        clips: list[tuple[int]]= None,
        vid_path: str = None,
        images_path: str = None,
        device: str="cpu", 
        sample_rate=1,
        transform: Optional[Callable] = None,
        sequential_matching_options: pycolmap.SequentialMatchingOptions = pycolmap.SequentialMatchingOptions(),
        mapping_options: pycolmap.IncrementalPipelineOptions = pycolmap.IncrementalPipelineOptions(),
        camera_model: str = "SIMPLE_RADIAL_FISHEYE",
        sequential_matching_kwargs: Optional[dict[str, Any]] = None,
        mapping_kwargs: Optional[dict[str, Any]] = None,
        geometric_verification: bool=True,
        num_threads: int = os.cpu_count(),
        output_path: str = "deepsfm_output",
        camera_mode: pycolmap.CameraMode = pycolmap.CameraMode.SINGLE,
    ):
        super().__init__(
            clips=clips, 
            vid_path=vid_path, 
            sample_rate=sample_rate, 
            transform=transform, 
            sequential_matching_options=sequential_matching_options, 
            mapping_options=mapping_options, 
            camera_model=camera_model, 
            sequential_matching_kwargs=sequential_matching_kwargs, 
            mapping_kwargs=mapping_kwargs,
            num_threads=num_threads,
            output_path=output_path,
            images_path=images_path,
        )
        self.device = device
        self.feature_extractor = feature_extractor.to(device)
        self.feature_matcher = feature_matcher.to(device)    
        self.pairgenerator = pairgenerator
        self.geometric_verification = geometric_verification
        self.camera_mode = camera_mode
        
        self.pairs = None # set later

        Path(self.output_path).mkdir(exist_ok=True, parents=True)
    
    @staticmethod
    def image_name_to_id(database_path) -> dict[str, str]:
        _db = pycolmap.Database(database_path)
        imgs = [x.todict() for x in _db.read_all_images()]
        names = {x['name']: x['image_id'] for x in imgs}

        return names
    
    @staticmethod
    def id_to_image_name(database_path) -> dict[str, str]:
        _db = pycolmap.Database(database_path)
        imgs = [x.todict() for x in _db.read_all_images()]
        names = {x['image_id']: x['name'] for x in imgs}

        return names

    def geometrically_verify_matches(self, database_path, pairs_file):
        with pycolmap.ostream():
            pycolmap.verify_matches(
                database_path,
                str(pairs_file),
                options=dict(ransac=dict(max_num_trials=20000, min_inlier_ratio=0.05)),
            )

    def get_and_write_pairs(self, database_path, pairs_path):
        id_to_img_name = self.id_to_image_name(database_path)
        pairs = self.pairgenerator(database=pycolmap.Database(database_path)).all_pairs()
        self.pairs = pairs
        
        # need to map id -> img name for pair txt file for geometric verification
        img_pairs = []
        for pair in pairs:
            id1, id2 = pair
            img1, img2 = id_to_img_name[id1], id_to_img_name[id2]
            img_pairs.append((img1, img2))
        
        with pairs_path.open("w") as f:
            f.write("\n".join(" ".join([i, j]) for i, j in deepcopy(img_pairs)))
        
        return img_pairs

    def extract_all_features(self, img_dir):
        print(f"Extracting features from all images...")
        files = [f for f in os.listdir(img_dir)]
        valid_files = [f for f in files if f.lower().endswith(tuple(self.VALID_IMG_EXTENSIONS))]

        if len(valid_files) == 0:
            raise ValueError(f"No valid image files found in {img_dir}, extensions must be one of {self.VALID_IMG_EXTENSIONS}")
        
        if len(valid_files) < len(files):
            print(f"Found {len(files) - len(valid_files)} invalid files in {img_dir}, ignoring these files.")

        feature_map = {}
        for file in tqdm(files):
            img = load_image(os.path.join(img_dir, file))
            img = img.to(self.device)
            feats = self.feature_extractor.extract(img)
            feature_map[file] = feats
            
        return feature_map

    def match_pair(self, feats0, feats1):
        matches01 = self.feature_matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension

        matches = matches01["matches"]
        return matches
    
    def write_all_keypoints(self, database_path, features):
        _db_colmap = pycolmap.Database(str(database_path))
        name_to_db_id_map = self.image_name_to_id(database_path)
        
        for img_name in features:
            img_id = name_to_db_id_map[img_name]
            
            keypoints = features[img_name]["keypoints"].cpu().squeeze(0).numpy().astype(np.float32)
            
            if not _db_colmap.exists_keypoints(img_id):
                _db_colmap.write_keypoints(img_id, keypoints)

        _db_colmap.close()

    def write_all_matches(
        self, 
        pairs,
        features: dict, 
        database_path: Path,
    ):
        name_to_db_id_map = self.image_name_to_id(database_path)
        _db_colmap = pycolmap.Database(str(database_path)) # pycolmap requires str 
        
        for f1, f2 in tqdm(pairs):
            matches = self.match_pair(features[f1], features[f2]).cpu().numpy().astype(np.uint32)

            id1 = name_to_db_id_map[f1]
            id2 = name_to_db_id_map[f2]
            
            if not _db_colmap.exists_matches(id1, id2):
                _db_colmap.write_matches(id1, id2, matches)

        _db_colmap.close()
        
    def write_images_to_db(self, img_dir: Path, database_path: Path) -> pycolmap.Database:
        database_path = str(database_path)
        img_dir = str(img_dir)
        
        # set up the empty tables on instantiation
        _db_colmap = pycolmap.Database(database_path)
        options = pycolmap.ImageReaderOptions()
        options.camera_model = self.camera_model
        pycolmap.import_images(
            database_path=database_path,
            image_path=img_dir,
            camera_mode=self.camera_mode,
            options=options,
        )
        _db_colmap.close()

    def register_single_clip(
        self,
        clip: Optional[tuple[int]] = None,
        mapping_options: pycolmap.IncrementalPipelineOptions = pycolmap.IncrementalPipelineOptions(),
    ):

        output_path = Path(self.output_path)
        output_path.mkdir(exist_ok=True, parents=True)
        
        if not self.images_path:
            image_dir = output_path / "images" / f"{clip[0]}_{clip[1]}"
            image_dir.mkdir(exist_ok=True, parents=True)
            print(f"Sampling clip {clip} from video...")
            self.sample_clip(clip, image_dir)
        else:
            image_dir = Path(self.images_path)

        database_path = output_path / "database.db"
        
        pairs_file = output_path / "pairs.txt"

        n_images = len(list(image_dir.glob("*")))  # potentially possible in future once we add filtering to sampling
        print(f"Registering {n_images} images")
        if n_images > 0:

            self.write_images_to_db(image_dir, database_path)
            pairs = self.get_and_write_pairs(database_path, pairs_file)
            
            features = self.extract_all_features(str(image_dir))
            self.write_all_keypoints(database_path, features)
            self.write_all_matches(pairs, features, database_path)

            # geometric verification of matched features
            if self.geometric_verification:
                self.geometrically_verify_matches(database_path, pairs_file)

            maps = pycolmap.incremental_mapping(  # triangulation & bundle adjustment
                database_path,
                image_dir,
                output_path,
                options=mapping_options,
                **self.mapping_kwargs,
            )

            shutil.rmtree(output_path)

            return maps

        return None

    def register(
        self,
    ) -> dict[str, pycolmap.Reconstruction]:
        """
        Register all video clips in parallel using COLMAP.
        """
        if self.vid_path is not None:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                for clip in self.clips:
                    future = executor.submit(
                        self.register_single_clip,
                        clip,
                        self.mapping_options,
                    )
                    futures.append(future)

                results = []
                for future in tqdm(futures, desc="Registering clips"):
                    results.append(future.result())
        else:
            results = self.register_single_clip(mapping_options=self.mapping_options)

        self.image_registration_results = results
        return results

