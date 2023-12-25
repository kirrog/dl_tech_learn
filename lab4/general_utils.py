from transformers import RobertaTokenizer, RobertaModel
import torch
from typing import Tuple, Dict
from typing import List, Union
import re
from collections import OrderedDict
import gc
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import sys 
import torch
import torch.nn as nn
from crf import CRF

# !!! Указан абсолютный путь !!!
sys.path.insert(0, "/home/dzigen/Desktop/ITMO/sem1/DLtech/dl_tech_learn/lab4")
from t_extraction_utils import TAGGED_NAMES, TLABEL_LIST, extract_tagged_features, TMODEL_STATE_PATH

###########################################

# !!! Указан абсолютный путь !!!
BASE_MODEL_DIR = '/home/dzigen/Desktop/medics2023/NLP_MODULE/models'
BASE_MODEL_PATH = BASE_MODEL_DIR + '/RuBioRoBERTa'

OUTPUT_DIR = './'

TOKEN_SEQ_LIMIT = 512
INPUTF_TEXT_COL = 'text'
OUTPUTF_TEXTID_COL = 'input_textid'
OUTPUTF_LABELS_COL = 'predicted_labels'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
LAYERS_TO_HOLD = ['23', '22', '21', 'pooler']

class RobertaCRF(nn.Module):
    def __init__(self, label_size, roberta_path=BASE_MODEL_PATH, device=DEVICE):
        super(RobertaCRF, self).__init__()

        self.device = DEVICE

        self.encoder = RobertaModel.from_pretrained(roberta_path).to(device)
        self.dropout = nn.Dropout(0.5)     
        self.linear = nn.Linear(self.encoder.config.hidden_size, label_size)
        self.crf = CRF(label_size)

        # Замораживаем часть слоёв бекбона
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.encoder.named_parameters():
            for hold_l in LAYERS_TO_HOLD:
                if hold_l in name:
                    param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None, mode='train'):
        embeddings = self.encoder(input_ids=input_ids, attention_mask= attention_mask)
        drop_out = self.dropout(embeddings.last_hidden_state)
        linear_out = self.linear(drop_out)

        if mode == 'train':
            crf_out = self.crf(linear_out, mask=attention_mask, 
                               labels=labels)
        elif mode == 'eval':
            crf_out = self.crf.viterbi_decode(linear_out, mask=attention_mask)
        else:
            raise KeyError
        
        return crf_out

MODEL_INFO = {
    'tagged': {
        'extract_func': extract_tagged_features,
        'labels': TLABEL_LIST,
        'model_class': RobertaCRF,
        'state_path': TMODEL_STATE_PATH,
        'names': TAGGED_NAMES
    }
}

###########################################

def init_model(model_type:str, base_model_path:str=BASE_MODEL_PATH, selected_device=None) -> Tuple[RobertaTokenizer, RobertaCRF]:

    global DEVICE
    if selected_device is not None:
        DEVICE = torch.device(selected_device)

    # 
    tokenizer = RobertaTokenizer.from_pretrained(base_model_path, 
                                                 add_prefix_space=True, use_fast=True)
    model = MODEL_INFO[model_type]['model_class'](len(MODEL_INFO[model_type]['labels']), base_model_path).to(DEVICE)

    #
    state_dict = torch.load(MODEL_INFO[model_type]['state_path'], map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    gc.collect()
    torch.cuda.empty_cache()

    return tokenizer, model

def make_extract(extract_type, input_file:str, tokenizer:RobertaTokenizer, 
                    model: RobertaCRF, 
                    save_output:bool=False, output_dir:str=OUTPUT_DIR) -> Tuple[int, str, pd.DataFrame]:

    extract_func = MODEL_INFO[extract_type]['extract_func']
    input_df = pd.read_csv(input_file, sep=';')

    # DEBUG
    # print(input_df.head())
    # print(input_df.shape)
    
    extracted_features = {f_name: [] for f_name in MODEL_INFO[extract_type]['names']}
    extracted_features[OUTPUTF_TEXTID_COL] = []
    extracted_features[OUTPUTF_LABELS_COL] = []

    for i in tqdm(range(input_df.shape[0])):
        fixed_text = preprocess_text(input_df[INPUTF_TEXT_COL][i])
        encoded_text_parts = encode_text(fixed_text, tokenizer)

        # print(encoded_text_parts)
        
        cur_features, pred_labels = extract_func(encoded_text_parts, tokenizer, model)
        cur_features[OUTPUTF_TEXTID_COL] = i
        extracted_features[OUTPUTF_LABELS_COL].append(pred_labels)

        for k, v in cur_features.items():
            extracted_features[k].append(v)
        
        gc.collect()
        torch.cuda.empty_cache()

    output_df = pd.DataFrame(extracted_features)

    if save_output:
        now = datetime.now()
        formated_time = now.strftime("%d%m%Y_%H%M%S")
        output_df.to_csv(f"{output_dir}/{extract_type}output({formated_time}).csv", sep=';', index=False)

    return output_df

def preprocess_text(text:str) -> str:
    return re.sub(" {2,}", " ",text)

def encode_text(text:str, tokenizer:RobertaTokenizer) -> List[Dict[str,torch.tensor]]:

    encoded_t = tokenizer(text)
    
    # Выравниваем последовательности 
    # до деления без остатка на TOKEN_SEQ_LIMIT
    for key in ['input_ids', 'attention_mask']:
        encoded_t[key] = encoded_t[key] + [0] * (TOKEN_SEQ_LIMIT - (len(encoded_t[key])%TOKEN_SEQ_LIMIT))
        
        # DEBUG
        #print(f"{key} len: {len(encoded_t[key])} | mod {len(encoded_t[key])%TOKEN_SEQ_LIMIT}")

    # Разбиваем токенизированный текста на части 
    # по TOKEN_SEQ_LIMIT токенов 
    encoded_text_parts = {key: torch.tensor(value).view(-1,TOKEN_SEQ_LIMIT) for key, value in encoded_t.items()}

    # DEBUG
    #for key in ['input_ids', 'attention_mask']:
    #    print(f"{key} size: {encoded_text_parts[key].size()}")

    return encoded_text_parts