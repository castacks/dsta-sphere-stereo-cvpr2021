#!/bin/bash

python3 python/main.py \
    --dataset_path /external/home/airlab/TempData/Ver1_3Cam_10142022/training \
    --conf sphere_stereo_calibration.json \
    --references_indices 0 \
    --visualize False \
    --min_dist 0.5 \
    --max_dist 50 \
    --candidate_count 32 \
    --matching_resolution 512 512 \
    --rgb_to_stitch_resolution 512 512 \
    --panorama_resolution 1024 512
