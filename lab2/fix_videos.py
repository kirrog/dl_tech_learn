import cv2
import os
from tqdm import tqdm
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

################################

RAW_VIDEOS_DIR = './datasets/raw_videos'
VIDEOS_DATASET_PATH = './datasets/videos_dataset.csv'
LABEL_NAMES = os.listdir(RAW_VIDEOS_DIR)
FIXED_VIDEOS_DIR = './datasets/fixed_videos'

video_links_df = pd.read_csv(VIDEOS_DATASET_PATH, sep=';')

################################

for name in LABEL_NAMES:
    raw_video_label_dir = f'{RAW_VIDEOS_DIR}/{name}'

    fixed_video_label_dir = f'{FIXED_VIDEOS_DIR}/{name}'
    if not os.path.exists(fixed_video_label_dir):
        os.mkdir(fixed_video_label_dir)

    label_videos = video_links_df[video_links_df['category'] == name]['links'].to_list()

    process = tqdm(label_videos)
    for video_file in process:
        process.set_description_str(f"{name} | {video_file}")
        from_file = f"{raw_video_label_dir}/{video_file}"
        to_file = f"{fixed_video_label_dir}/{video_file}"

        cmd = f"ffmpeg -i {from_file} -c:v mpeg4 {to_file}"
        os.system(cmd)