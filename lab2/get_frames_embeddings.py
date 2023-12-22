from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
import cv2
from random import random
import pandas as pd

#################################

ROOT_DIR = '/home/ubuntu/DL/'
M2F_MODEL_NAME = ROOT_DIR + "pretrained_model/facebook-m2f_swin_large"
FRAMES_DIR = ROOT_DIR + 'frames'
LABEL_NAMES = os.listdir(FRAMES_DIR)
DEVICE = 'cuda'

#################################

class M2Fencoder(nn.Module):
    def __init__(self):
        super(M2Fencoder, self).__init__()
        
        #
        m2f = Mask2FormerForUniversalSegmentation.from_pretrained(M2F_MODEL_NAME)
        m2f.requires_grad_(False)
        self.bb_features = 1536

        # M2F backbone
        self.embeddings = m2f.model.pixel_level_module.encoder.embeddings
        self.encoder = m2f.model.pixel_level_module.encoder.encoder
        self.layernorm = nn.LayerNorm(self.bb_features)
        self.pooler = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        embedding_output, input_dimensions = self.embeddings(x)
        encoder_outputs = self.encoder(embedding_output, input_dimensions)
        sequence_output = encoder_outputs.last_hidden_state

        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output.transpose(1, 2))

        return pooled_output

class ImageDataset(Dataset):
    def __init__(self, images_names, images_dir, transform):
        self.images_dir = images_dir
        self.image_names = images_names
        self.transform_part1 = A.Compose([A.augmentations.dropout.coarse_dropout.CoarseDropout(max_height=16, max_width=16, max_holes=16)])
        self.transform_part2 = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_filepath = f'{self.images_dir}/{self.image_names[index]}'

        image = cv2.imread(image_filepath)
        image_tensor1 = self.transform_part1(image=image, return_tensors="pt")['image']
        image_tensor2 = self.transform_part2(image_tensor1, return_tensors="pt")['pixel_values']

        return image_tensor2
    
#################################
print("LOAD MODEL")
    
m2f_model = M2Fencoder().to(DEVICE)
m2f_processor = AutoImageProcessor.from_pretrained(M2F_MODEL_NAME)

#################################

dataset = {}
for frames_dir_name in LABEL_NAMES:
    cur_dir_path = f"{FRAMES_DIR}/{frames_dir_name}"
    frame_names = os.listdir(cur_dir_path)
    dataset[frames_dir_name] =  ImageDataset(frame_names, cur_dir_path, m2f_processor)
    print(frames_dir_name, len(dataset[frames_dir_name]))

#################################
print("START FRAME CONVERTION")

embeddings_size = m2f_model.bb_features
df_columns = [f"x{i} "for i in range(embeddings_size)] + ['labels']
df = pd.DataFrame(columns=df_columns)

for label_name in LABEL_NAMES[3:]:
    process = tqdm(range(len(dataset[label_name])))
    tmp_df = []
    for i in process:
        process.set_description_str(label_name)
        output = m2f_model(dataset[label_name][i].to(DEVICE))
        image_embedding = output.view(-1, 1536).detach().cpu().numpy().tolist()
        tmp_df.append(image_embedding[0] + [label_name])
    
    tmp_df = pd.DataFrame(tmp_df,columns=df_columns)
    df = pd.concat([df, tmp_df]).reset_index(drop=True)
    
    df.to_csv("frames_embeddings.csv", index=False, sep=';')
    print("cur df size: ", df.shape)