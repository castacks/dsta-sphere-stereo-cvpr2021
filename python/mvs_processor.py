import cv2
from joblib import Parallel, delayed
import json
import numpy as np
import os
import time

import torch

import cupy
from .depth_estimation import RGBD_Estimator
from .utils import (
    parse_json_calib )


class MVSProcessor(object):
    def __init__(self, 
        min_dist, max_dist,
        candidate_count,
        matching_resolution,
        rgb_to_stitch_resolution,
        panorama_resolution,
        references_indices,
        sigma_i, sigma_s):

        super().__init__()

        self.min_dist = min_dist
        self.max_dist = max_dist
        self.candidate_count = candidate_count
        self.matching_resolution = matching_resolution
        self.rgb_to_stitch_resolution = rgb_to_stitch_resolution
        self.panorama_resolution  =panorama_resolution
        self.references_indices = references_indices
        self.sigma_i = sigma_i
        self.sigma_s = sigma_s

        self.calibrations = None # List

        self.reprojection_viewpoint = None # 3-Tensor

        self.rgbd_estimator = None

        self.initialized = False

    @property
    def device(self):
        return 'cuda:0'

    def parse_calibration(self, fn):
        with open(fn, 'r') as fp:
            raw_calibration = json.load(fp)

        self.calibrations = parse_json_calib(
            raw_calibration,
            self.matching_resolution,
            self.device )

    def read_masks(self, mask_dir):
        masks = []
        for cam_index in range(len(self.calibrations)):
            if os.path.isfile(os.path.join(mask_dir, "cam" + str(cam_index)) + "/" + "mask.png"):
                mask = cv2.imread(os.path.join(mask_dir, "cam" + str(cam_index)) + "/" + "mask.png", 
                                    cv2.IMREAD_UNCHANGED)
                mask = cv2.resize(mask, tuple(self.matching_resolution), cv2.INTER_AREA)
                masks.append(torch.tensor(mask, device=self.device, dtype=torch.float32).unsqueeze(0)/255)
            else:
                masks.append(torch.ones(self.matching_resolution, device=self.device).unsqueeze(0))

        return masks

    def initialize(self, calib_fn, mask_dir):
        self.parse_calibration(calib_fn)

        # Reference viewpoint for the estimated RGB-D panorama is the center of the references
        self.reprojection_viewpoint = torch.zeros([3], device=self.device)
        for references_index in self.references_indices:
            self.reprojection_viewpoint += self.calibrations[references_index].rt[:3, 3] / len(self.references_indices)

        # Read masks
        masks = self.read_masks(mask_dir)
        
        # Initialize distance estimator and stitcher
        self.rgbd_estimator = RGBD_Estimator(
            self.calibrations, 
            self.min_dist, self.max_dist, 
            self.candidate_count, 
            self.references_indices, 
            self.reprojection_viewpoint, 
            masks, 
            self.matching_resolution, 
            self.rgb_to_stitch_resolution, 
            self.panorama_resolution, 
            self.sigma_i, self.sigma_s, 
            self.device)

        self.initialized = True

    def pre_process(self, images):
        images_to_stitch = []
        images_to_match = []

        for cam_index, calibration in enumerate(self.calibrations):
            valid_frame = True
            image = images[cam_index]
            
            print(image.shape, (calibration.original_resolution[1], calibration.original_resolution[0], 3))
            if image.shape == (calibration.original_resolution[1], calibration.original_resolution[0], 3):
                # Map all types range to [0, 255] as float32
                if image.dtype == np.uint8:
                    image = image.astype(np.float32)
                elif image.dtype == np.uint16:
                    image = image.astype(np.float32) / 255
                elif image.dtype == np.float32:
                    if np.max(image) > 1:
                        image = np.clip(image, 0, 1)
                    image = image * 255
                else:
                    print(f"Invalide image type {image.dtype}")
                    valid_frame = False
            else:
                valid_frame = False


            if valid_frame:
                # Keep references at higher resolution for stitching
                if cam_index in self.references_indices:
                    image_to_stitch = cv2.resize(image, tuple(self.rgb_to_stitch_resolution), cv2.INTER_AREA)
                    images_to_stitch.append(image_to_stitch)
                # Resize for matching and distance estimation
                image_to_match = cv2.resize(image, tuple(self.matching_resolution), cv2.INTER_AREA)
                images_to_match.append(image_to_match)
        
        return {
            "images_to_match": images_to_match, 
            "images_to_stitch": images_to_stitch, 
            "is_valid": valid_frame }

    def __call__(self, images):
        # Pre-processing.
        pre_processed_images = self.pre_process(images)

        if not pre_processed_images['is_valid']:
            return

        start_time = time.time()
        fisheye_images = [
            torch.tensor(fisheye_image, device=self.device) 
            for fisheye_image in pre_processed_images['images_to_match']]
        
        reference_fisheye_images = [
            torch.tensor(reference_fisheye_image, device=self.device) 
            for reference_fisheye_image in pre_processed_images['images_to_stitch']]
        rgb, distance = \
            self.rgbd_estimator.estimate_RGBD_panorama(
                fisheye_images, reference_fisheye_images)
        
        res = {
            "rgb": rgb.cpu().numpy(), 
            "inv_distance": 1 / distance.cpu().numpy() }
        end_time = time.time()
        print(f'Processing time: {end_time - start_time}s')

        return res
