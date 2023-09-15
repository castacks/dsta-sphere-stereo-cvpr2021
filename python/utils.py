"""
=======================================================================
General Information
-------------------
This is a GPU-based python implementation of the following paper:
Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images
Andreas Meuleman, Hyeonjoong Jang, Daniel S. Jeon, Min H. Kim
Proc. IEEE Computer Vision and Pattern Recognition (CVPR 2021, Oral)
Visit our project http://vclab.kaist.ac.kr/cvpr2021p1/ for more details.

Please cite this paper if you use this code in an academic publication.
Bibtex: 
@InProceedings{Meuleman_2021_CVPR,
    author = {Andreas Meuleman and Hyeonjoong Jang and Daniel S. Jeon and Min H. Kim},
    title = {Real-Time Sphere Sweeping Stereo from Multiview Fisheye Images},
    booktitle = {CVPR},
    month = {June},
    year = {2021}
}
==========================================================================
License Information
-------------------
CC BY-NC-SA 3.0
Andreas Meuleman and Min H. Kim have developed this software and related documentation (the "Software"); confidential use in source form of the Software, without modification, is permitted provided that the following conditions are met:
Neither the name of the copyright holder nor the names of any contributors may be used to endorse or promote products derived from the Software without specific prior written permission.
The use of the software is for Non-Commercial Purposes only. As used in this Agreement, "Non-Commercial Purpose" means for the purpose of education or research in a non-commercial organisation only. "Non-Commercial Purpose" excludes, without limitation, any use of the Software for, as part of, or in any way in connection with a product (including software) or service which is sold, offered for sale, licensed, leased, published, loaned or rented. If you require a license for a use excluded by this agreement, please email [minhkim@kaist.ac.kr].
Warranty: KAIST-VCLAB MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. KAIST-VCLAB SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
Please refer to license.txt for more details.
=======================================================================
"""
import math
import torch
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import cv2 
import os.path 
import warnings
import numpy as np
import re
# from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from .image_io import ( 
            read_compressed_float,
            write_compressed_float, 
            write_float_image_fixed_normalization,
            write_float_image_normalized)

class Calibration:
    def __init__(self, original_resolution, principal, fl, xi, alpha, rt, matching_scale):
        """
        Args:
            original_resolution, principal, fl, xi, alpha: Double sphere intrinsics
            rt: [4, 4] camera pose
            matching_scale: [2] Scale to apply to the resolution, principal and fl 
                when working with images resized to the matching resolution
        """
        self.original_resolution = original_resolution
        self.principal = principal
        self.fl = fl
        self.xi = xi
        self.alpha = alpha
        self.rt = rt
        self.matching_scale = matching_scale 

def unproject(uv,  calib):
    """
    Unproject pixels to the unit sphere following the The Double Sphere Camera Model (https://arxiv.org/abs/1807.08957)
    Apply the calib.matching_scale to fit the distance estimation resolution
    """

    m_xy = (uv - calib.principal * calib.matching_scale) / (calib.fl * calib.matching_scale)

    r2 = torch.sum(m_xy**2, dim=-1, keepdim=True)
    m_z = ((1 - calib.alpha**2 * r2) 
            / (calib.alpha * torch.sqrt(torch.clamp(1 - (2 * calib.alpha - 1) * r2, min=0)) + 1 - calib.alpha))

    point = torch.cat([m_xy, m_z], dim=-1)
    point = ((m_z * calib.xi + torch.sqrt(m_z**2 + (1 - calib.xi**2) * r2)) / (m_z**2 + r2)) * point
    point[..., 2] -= calib.xi

    valid = (1 - (2 * calib.alpha - 1) * r2 >= 0)

    return point, valid[..., 0]

def project(point, calib):
    """
    Project a point in space to pixel coordinates (https://arxiv.org/abs/1807.08957)
    Apply the calib.matching_scale to fit the distance estimation resolution
    """

    d1 = torch.norm(point, dim=-1, keepdim=True)

    c = calib.xi * d1 + point[..., 2:3]
    d2 = torch.norm(torch.cat([point[..., :2], c], dim=-1), dim=-1, keepdim=True)
    norm = calib.alpha * d2 + (1 - calib.alpha) * c
    
    if(calib.alpha > 0.5):
        w1 = (1 - calib.alpha) / calib.alpha 
    else: 
        w1 = calib.alpha / (1 - calib.alpha)
    w2 = (w1 + calib.xi) / math.sqrt(2 * w1 * calib.xi + calib.xi**2 + 1)

    valid = point[..., 2:3] > - w2 * d1
    uv = (calib.fl * calib.matching_scale * point[..., :2]) / norm + calib.principal * calib.matching_scale

    return uv, valid[..., 0]

def parse_json_calib(raw_calibration, matching_resolution, device):
    """
    Parse basalt-formated calibration file (https://gitlab.com/VladyslavUsenko/basalt/-/blob/master/doc/Calibration.md)
    Args:
        matching_resolution: [2] Resolution at which the fisheye images will be resize for distance estimation.
            It is used to obtain the matching_scale component of the calibration.
    """
    calibrations = []
    for extrinsics, intrinsics, original_resolution \
            in zip(raw_calibration['T_imu_cam'], raw_calibration['intrinsics'], raw_calibration['resolution']):

        cam_intrinsics = intrinsics['intrinsics']

        # SciPy 1.2.3 does not support as_matrix()
        # r = R.from_quat([
        #     extrinsics['qx'],
        #     extrinsics['qy'],
        #     extrinsics['qz'],
        #     extrinsics['qw']
        # ])
        r = Quaternion(
            extrinsics['qw'],
            extrinsics['qx'],
            extrinsics['qy'],
            extrinsics['qz']
        )

        t = torch.tensor([
            extrinsics['px'],
            extrinsics['py'],
            extrinsics['pz']
        ], device=device)

        rt = torch.eye(4, device=device)
        # SciPy 1.2.3 does not support as_matrix()
        # rt[:3, :3] = torch.tensor(r.as_matrix(), device=device)
        rt[:3, :3] = torch.tensor(r.rotation_matrix, device=device)
        rt[:3, 3] = t

        if intrinsics["camera_type"] == "ds":
            calibrations.append(Calibration(
                original_resolution,
                torch.tensor([cam_intrinsics['cx'], cam_intrinsics['cy']], device=device),
                torch.tensor([cam_intrinsics['fx'], cam_intrinsics['fy']], device=device),
                cam_intrinsics['xi'],
                cam_intrinsics['alpha'],
                rt,
                torch.tensor([
                        matching_resolution[0] / original_resolution[0],
                        matching_resolution[1] / original_resolution[1]
                    ], device=device)
            ))
        else: 
            raise Exception("Unexpected camera model. The current implementation only support double sphere.")

    return calibrations

def rgb2yCbCr(rgb):
    rgb = rgb.float()
    yuv = torch.zeros_like(rgb)

    yuv[:, :, 0] = torch.clamp(16  + 0.1826 * rgb[:, :, 0] + 0.6142 * rgb[:, :, 1] + 0.062  * rgb[:, :, 2]
                               , min=16, max=235)
    yuv[:, :, 1] = torch.clamp(128 - 0.1006 * rgb[:, :, 0] - 0.3386 * rgb[:, :, 1] + 0.4392 * rgb[:, :, 2] 
                               , min=16, max=240)
    yuv[:, :, 2] = torch.clamp(128 + 0.4392 * rgb[:, :, 0] - 0.3989 * rgb[:, :, 1] - 0.0403 * rgb[:, :, 2]
                               , min=16, max=240)

    return yuv

def read_input_images(filename, dataset_path, matching_resolution, rgb_to_stitch_resolution, 
                      calibrations, references_indices):
    """
    Read and resize fisheye images 
    """
    images_to_match = []
    images_to_stitch = []
    valid_frame = True
    # Read input image for each camera
    for cam_index, calibration in enumerate(calibrations):
        file_path = os.path.join(dataset_path, "cam" + str(cam_index)) + "/" + filename
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        # Type and innapropriate file handling
        if image is not None:
            if image.shape == (calibration.original_resolution[1], calibration.original_resolution[0], 3):
                # Map all types range to [0, 255] as float32
                if image.dtype == np.uint8:
                    image = image.astype(np.float32)
                elif image.dtype == np.uint16:
                    image = image.astype(np.float32) / 255
                elif image.dtype == np.float32:
                    if np.max(image) > 1:
                        image = np.clip(image, 0, 1)
                        warnings.warn("Image has out-of-range float values for file " 
                                      + file_path + ". Clipped for processing.")
                    image = image * 255
                else:
                    warnings.warn("Invalide image type for file " + file_path)
                    valid_frame = False
            else:
                warnings.warn("Invalid image size / channels for file " + file_path)
                valid_frame = False

        else:
            warnings.warn("Cannot read image for file " + file_path)
            valid_frame = False
        
        if valid_frame:
            # Keep references at higher resolution for stitching
            if cam_index in references_indices:
                image_to_stitch = cv2.resize(image, tuple(rgb_to_stitch_resolution), cv2.INTER_AREA)
                images_to_stitch.append(image_to_stitch)
            # Resize for matching and distance estimation
            image_to_match = cv2.resize(image, tuple(matching_resolution), cv2.INTER_AREA)
            images_to_match.append(image_to_match)

    return {"images_to_match": images_to_match, "images_to_stitch": images_to_stitch, "is_valid": valid_frame}

class GTLoader(object):
    def __init__(self, folder_name):
        '''
        folder_name: Folder name of the ground truth.
        '''
        super().__init__()
        
        self.folder_name = folder_name
        
    def read_rgb(self, fn):
        rgb = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        assert rgb is not None, f'Read {fn} failed. '
        return rgb
    
    def read_dist(self, fn):
        return read_compressed_float(fn)
    
    def read_inv_dist(self, fn):
        dist = self.read_dist(fn)
        return 1.0 / dist
    
    def load(self, dataset_root: str, fisheye_fn: str):
        '''
        Load the ground truth based on the fisheye image filename.
        '''
        raise NotImplementedError()
    
class SphereStereoGTLoader(GTLoader):
    def __init__(self):
        super().__init__('gt')
        
    def load(self, dataset_root: str, fisheye_fn: str):
        '''
        Load the ground truth based on the fisheye image filename.
        '''
        
        # Extract the index from the filename.
        index = os.path.splitext(fisheye_fn)[0]
        
        # Read the ground truth.
        gt_rgb = self.read_rgb( os.path.join( dataset_root, self.folder_name, f'rgb_{index}.png'))
        gt_inv_dist = self.read_inv_dist( os.path.join( dataset_root, self.folder_name, f'distance_{index}.png' ) )
        
        return gt_rgb, gt_inv_dist

class DSTAGTLoader(GTLoader):
    def __init__(self):
        super().__init__('rig')
        
        self.index_extractor = re.compile(r'(\d+)_CubeScene.png')
        
    def load(self, dataset_root: str, fisheye_fn: str):
        '''
        Load the ground truth based on the fisheye image filename.
        '''
        
        # Extract the index from the filename.
        matches = self.index_extractor.search(fisheye_fn)
        assert matches is not None, f'Fisheye image filename {fisheye_fn} does not have leading index. '
        index = matches.group(1)
        
        # Read the ground truth.
        gt_rgb = self.read_rgb( os.path.join( dataset_root, self.folder_name, f'{index}_CubeScene.png'))
        gt_inv_dist = self.read_inv_dist( os.path.join( dataset_root, self.folder_name, f'{index}_CubeDistance.png' ) )
        
        return gt_rgb, gt_inv_dist

GT_LOADERS = dict(
    SphereStereoGTLoader=SphereStereoGTLoader,
    DSTAGTLoader=DSTAGTLoader
)

def evaluate_rgbd_panorama(rgbd_panoramas, filename, dataset_path, gt_loader, bad_px_ratio_thresholds, panorama_resolution):
    """
    Read ground truth in <dataset_path>/gt/
    Compute PSNR, SSIM, MAE, RMSE and bad pixel ratio on an RGB-D panorama.
    Args:
        rgbd_panoramas: dict[filename: dict['rgb': [rows, cols, 3] uint8, 'inv_distance': [rows, cols] float32]]
    """
    try:
        rgbd_panorama = rgbd_panoramas[filename]

        # read_name = os.path.splitext(filename)[0]
        # gt_rgb = cv2.imread(os.path.join(dataset_path, "gt/rgb_" + read_name + ".png"), cv2.IMREAD_UNCHANGED)
        # gt_distance = cv2.imread(os.path.join(dataset_path, "gt/inv_distance_" + read_name + ".exr"), 
        #                          cv2.IMREAD_UNCHANGED)

        evaluated_rgb = rgbd_panorama["rgb"]
        evaluated_distance = rgbd_panorama["inv_distance"]
        
        gt_rgb, gt_distance = gt_loader.load( dataset_path, filename )

        if(gt_rgb is not None and gt_rgb is not None 
                and gt_rgb.dtype == evaluated_rgb.dtype and gt_distance.dtype == evaluated_distance.dtype):
            
            gt_rgb = cv2.resize(gt_rgb, tuple(panorama_resolution), cv2.INTER_AREA)
            gt_distance = cv2.resize(gt_distance, tuple(panorama_resolution), cv2.INTER_AREA)
            
            gt_rgb = gt_rgb.astype(np.float32) / 255
            evaluated_rgb = evaluated_rgb.astype(np.float32) / 255
            # scikit-image 0.16.2 needs numpy 1.20.0 which is not available at the moment.
            # ssim = structural_similarity(gt_rgb, evaluated_rgb, multichannel=True)
            # psnr = peak_signal_noise_ratio(gt_rgb, evaluated_rgb)
            raise Exception('scikit-image 0.16.2 needs numpy 1.20.0 which is not available at the moment. ')

            err = np.abs(evaluated_distance - gt_distance)
            mae = np.sum(err) / err.size
            
            err2 = err * err
            rmse = np.sqrt(np.sum(err2) / err2.size)
            
            bad_px_ratios = []
            for bad_px_ratio_threshold in bad_px_ratio_thresholds:
                bad_px_ratios.append(100 * np.sum(err > bad_px_ratio_threshold) / err.size)

            return {"ssim": ssim, "psnr": psnr, "mae": mae, "rmse": rmse, "bad_px_ratios": bad_px_ratios}
        else:
            warnings.warn("Invalid ground truth for file " + filename + ". Will be ignored for evaluation")
            return None

    except KeyError:
        warnings.warn("Invalid ground truth for file " + filename)
        return None

def save_rgbd_panorama(rgbd_panoramas, filename, dataset_path):
    try:
        rgbd_panorama = rgbd_panoramas[filename]
        save_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(dataset_path, "output/rgb_" + save_name + ".png"), rgbd_panorama["rgb"])
        # cv2.imwrite(os.path.join(dataset_path, "output/inv_distance_" + save_name + ".exr"), 
        #             rgbd_panorama["inv_distance"])
        
        inv_dist = rgbd_panorama["inv_distance"]
        
        write_compressed_float(os.path.join(dataset_path, "output/distance_" + save_name + ".png"), 
                    1.0 / inv_dist )
        
        invalid_mask = np.logical_not(np.isfinite(inv_dist))
        inv_dist[invalid_mask] = 1.0 / 100
        # write_float_image_normalized(os.path.join(dataset_path, "output/distance_" + save_name + "_vis.png"), 
        #             1.0 / rgbd_panorama["inv_distance"])
        write_float_image_normalized(os.path.join(dataset_path, "output/inv_distance_" + save_name + "_vis.png"), 
                    inv_dist )
    except KeyError:
        pass