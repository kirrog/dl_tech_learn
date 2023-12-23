import pandas as pd 
import cv2
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import os
from catboost import CatBoostClassifier, Pool
from math import floor, ceil
import torch
from tqdm import tqdm
from time import time
from sklearn.metrics import classification_report

###################################

ROOT_DIR = '/home/dzigen/Desktop/ITMO/sem1/DLtech/dl_tech_learn/lab2/'

LABELS = ['travel','art_music','food','history']
ID2LABEL = {i:v for i, v in enumerate(LABELS)}
LABEL2ID = {v:i for i, v in enumerate(LABELS)}

EMBEDDER_PATH = ROOT_DIR + 'pretrained_model/facebook-m2f_swin_large'
EMBEDDER_PROCESSOR_PATH = ROOT_DIR + 'pretrained_model/facebook-m2f_swin_large'
FRAME_CLASSIFIER_PATH = ROOT_DIR + 'frame_classifier_model'
START_PROP_OFFSET = 5
END_PROP_OFFSET = 10
EMBEDDINGS_SIZE = 1536
DEVICE = 'cuda'

VIDEOS_PER_LABEL = 5
VIDEO_DATASET_PATH = ROOT_DIR + 'videos_dataset.csv'
VIDEO_DIR = ROOT_DIR + 'fixed_videos/'
PREDICTIONS_FILE = ROOT_DIR + 'video_label_predictions.csv'

###################################

class EmbedderProcessor:
    def __init__(self, processor_path=EMBEDDER_PROCESSOR_PATH):
        self.transform_part1 = A.Compose([A.augmentations.dropout.coarse_dropout.CoarseDropout(
            max_height=16, max_width=16, max_holes=16)])
        self.transform_part2 = AutoImageProcessor.from_pretrained(processor_path)

    def transform(self, frame):
        image_tensor1 = self.transform_part1(image=frame, return_tensors="pt")['image']
        image_tensor2 = self.transform_part2(image_tensor1, return_tensors="pt")['pixel_values']
        return image_tensor2

class Embedder(nn.Module):
    def __init__(self,embedder_path=EMBEDDER_PATH):
        super(Embedder, self).__init__()
        
        #
        m2f = Mask2FormerForUniversalSegmentation.from_pretrained(embedder_path)
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

def classify_video(video_path, emb_model, emb_processor, fcls_model):
    print("Loading video...", end='')
    s_time = time()
    video_object = load_video(video_path)
    e_time = time()
    print(round(e_time - s_time, 3), "sec")


    print("Cuting head and tail...", end='')
    s_time = time()
    selected_frame_ids = reduce_video(video_object)
    label_freqs = {label: 0 for label in LABELS}
    frames_amount = len(selected_frame_ids)
    e_time = time()
    print(round(e_time - s_time, 3), "sec")

    print("Predicting labels for selected frames...")
    process = tqdm(selected_frame_ids)
    for frame_id in process:
        ret, frame = get_frame_by_id(video_object, frame_id)
        frame_emb = get_frame_embedding(frame, emb_model, emb_processor)
        pred_label = classify_frame(frame_emb, fcls_model)
        label_freqs[pred_label] += 1
        process.set_description_str(pred_label)

    print("Calculating labels proportion...", end='')
    s_time = time()
    label_probs = {k: round(v / frames_amount, 3) for k,v in label_freqs.items()}
    frequent_label = sorted(label_probs.items(), key=lambda v: v[1], reverse=True)[0]
    e_time = time()
    print(round(e_time - s_time, 3), "sec")

    print("Done!")
    return frequent_label[0], label_probs, selected_frame_ids

def load_video(video_path):
    return cv2.VideoCapture(video_path)

def get_frame_by_id(video_object, frame_id):
    video_object.set(1, frame_id)
    return video_object.read()

def get_frame_embedding(frame, emb_model, emb_processor):
    transformed_frame = emb_processor.transform(frame)
    with torch.no_grad():
        output = emb_model(transformed_frame.to(DEVICE))
    frame_embedding = output.view(-1, EMBEDDINGS_SIZE).detach().cpu().numpy().tolist()[0]

    return frame_embedding


def classify_frame(frame_emb, fcls_model):
    output = fcls_model.predict(frame_emb)
    return ID2LABEL[output[0]]

def init_models(emb_model_path, emb_processor_path, fcls_model_path):
    emb_model = Embedder(emb_model_path).to(DEVICE)
    emb_model.eval()

    emb_processor = EmbedderProcessor(emb_processor_path)
    
    fcls_model = CatBoostClassifier()
    fcls_model.load_model(fcls_model_path)

    return emb_model, emb_processor, fcls_model

def reduce_video(video_object):
    frames_amount = int(video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_id = list(range(frames_amount))
    fps = ceil(video_object.get(cv2.CAP_PROP_FPS))
    print(f"[before reduce: {frames_amount}]...",end='')

    filtered_frames = frames_id[::fps]
    print(f"[after fps filter: {len(filtered_frames)}]...",end='')

    head_offset = (len(filtered_frames) * START_PROP_OFFSET) // 100
    tail_offset = (len(filtered_frames) * END_PROP_OFFSET) // 100

    return filtered_frames[head_offset:-tail_offset]

###################################

videos_df = pd.read_csv(VIDEO_DATASET_PATH,sep=';')
emb_model, emb_processor, fcls_model = init_models(EMBEDDER_PATH, EMBEDDER_PROCESSOR_PATH, FRAME_CLASSIFIER_PATH)

###################################

info = []
for label in LABELS:
    selected_links = videos_df[videos_df['category'] == label]['links'].to_list()[:VIDEOS_PER_LABEL]

    for i, link in enumerate(selected_links):
        print(f"{label} | {link} | {i} / {len(selected_links)}")
        s_time = time()
        path_to_cur_video = f"{VIDEO_DIR}{label}/{link}"
        pred_label, label_distr, selected_frame_ids = classify_video(path_to_cur_video, emb_model, emb_processor, fcls_model)
        e_time = time()

        print("Output: ", pred_label, label_distr)

        info.append([
            link, label, pred_label,
            label_distr, round(e_time-s_time, 3), 
            len(selected_frame_ids)
        ])

###################################

test_df = pd.DataFrame(info, columns=['links','refs','preds','distrs',
                                      'elapsed_time (sec)', 
                                      'classified frames amount'])
test_df.to_csv(PREDICTIONS_FILE, sep=';',index=False)

###################################

y_true = list(map(lambda v: LABEL2ID[v], test_df['refs']))
y_pred = list(map(lambda v: LABEL2ID[v], test_df['preds']))
print(classification_report(y_true, y_pred, target_names=LABELS))