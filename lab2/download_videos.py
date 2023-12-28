from collections import Counter
import pandas as pd
import json
import os
from tqdm import tqdm

###################################

CLEANED_DATA_FILE = './datasets/cleaned_data.csv'
VIDEOS_PER_LABEL = 900

data_df = pd.read_csv(CLEANED_DATA_FILE)
print(Counter(data_df['category'].to_list()))

LABEL_NAMES = list(data_df['category'].unique())
print(LABEL_NAMES)

SELECTED_LINKS = {name: list(set(data_df[data_df['category'] == name]['link'].to_list()))[:VIDEOS_PER_LABEL] for name in LABEL_NAMES}

for k,v in SELECTED_LINKS.items():
	del SELECTED_LINKS[k][SELECTED_LINKS[k].index('#NAME?')]
	print(f"{k}: {len(v)}")

# Пересечения множеств
print(set(SELECTED_LINKS['travel']).intersection(set(SELECTED_LINKS['food'])))
print(set(SELECTED_LINKS['travel']).intersection(set(SELECTED_LINKS['art_music'])))
print(set(SELECTED_LINKS['travel']).intersection(set(SELECTED_LINKS['history'])))
print(set(SELECTED_LINKS['food']).intersection(set(SELECTED_LINKS['art_music'])))
print(set(SELECTED_LINKS['food']).intersection(set(SELECTED_LINKS['history'])))
print(set(SELECTED_LINKS['art_music']).intersection(set(SELECTED_LINKS['history'])))

###################################
    
OUTPUT_PATH = './datasets/raw_videos'
if not os.path.exists(OUTPUT_PATH):
	os.mkdir(OUTPUT_PATH)
    
###################################

downloading_label_links = list(SELECTED_LINKS.keys())[-1::-1][:2]
for video_labels in downloading_label_links:
	process = tqdm(SELECTED_LINKS[video_labels])
	
	labels_dir = f"{OUTPUT_PATH}/{video_labels}"
	if not os.path.exists(labels_dir):
		os.mkdir(labels_dir)
	for video_link in process:
		video_path = f"{labels_dir}/{video_link}.mp4"
		if os.path.exists(video_path):
			continue
		
		process.set_postfix_str(f"{video_labels}: {video_link}")
		os.system(f'yt-dlp -o {video_path} --format worstvideo[width=426][height=240] https://www.youtube.com/watch?v=' + video_link)
