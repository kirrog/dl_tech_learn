from transformers import RobertaForTokenClassification, RobertaTokenizer
import torch
from typing import Tuple, Dict
from typing_extensions import List
import gc
import re
import numpy as np
import json
from functools import reduce
import pandas as pd

# !!! Указан абсолютный путь !!!
TAGGED_MODEL_DIR = '/home/dzigen/Desktop/ITMO/sem1/DLtech/dl_tech_learn/lab4/saved_model'
TMODEL_STATE_PATH = TAGGED_MODEL_DIR + '/ner_model'

TAGGED_NAMES = ["КТР","ЧСС","ТВП","ТРИКУСП РЕГУРГ","ХГЧ","PAPP-A","АФП", "Ингибин А",
               "св эстрадиол","Возраст","Вес","Рост","ИМТ","ФК"]

TLABEL_LIST = ['O']
LABELNUM2NAME = {}
for i, f_name in enumerate(TAGGED_NAMES):
    f_b, f_i = f"B-L{i+1}", f"I-L{i+1}"
    TLABEL_LIST.append(f_b)
    TLABEL_LIST.append(f_i)
    LABELNUM2NAME[i+1] = f_name

LABEL2ID = {label: i for i, label in enumerate(TLABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(TLABEL_LIST)}

def extract_tagged_features(encoded_text_parts:Dict[str,torch.tensor], 
                            tokenizer:RobertaTokenizer, model:RobertaForTokenClassification) -> Dict[str,bool]:

    features = {name: [] for name in TAGGED_NAMES}
    
    batch = {k:v.to(model.device) for k, v in encoded_text_parts.items()}
    with torch.no_grad():
        output = model(batch['input_ids'],batch['attention_mask'], mode='eval')

    predictions = output
    labels = [[ID2LABEL[p] for p in pred_part] for pred_part in predictions ]
    token_ids = batch['input_ids'].to('cpu').numpy().tolist()

    #
    attention_end = [v.tolist().index(0) for v in batch['attention_mask']]
    accum_tokens, seq_cls, prev_tag = [], -1, 'O'
    for i, label_part in enumerate(labels):
        for j, l in enumerate(label_part):
            if j == attention_end[i]:
                accum_tokens, seq_cls, prev_tag = [], -1, 'O'
                break

            if l.startswith('B-'):
                if prev_tag == 'O':
                    accum_tokens.append(token_ids[i][j])
                    prev_tag, seq_cls = 'B', int(l.split('L')[1])

                elif prev_tag == 'B':
                    cur_cls = int(l.split('L')[1])
                    if cur_cls != seq_cls:
                        features[LABELNUM2NAME[seq_cls]].append(tokenizer.decode(accum_tokens))
                        accum_tokens, seq_cls, prev_tag = [token_ids[i][j]], cur_cls, 'B'
                    else:
                        accum_tokens.append(token_ids[i][j])

                elif prev_tag == 'I':
                    if seq_cls == -1:
                        accum_tokens, seq_cls, prev_tag = [token_ids[i][j]], cur_cls, 'B'
                    else:
                        features[LABELNUM2NAME[seq_cls]].append(tokenizer.decode(accum_tokens))
                        cur_cls = int(l.split('L')[1])
                        accum_tokens, seq_cls, prev_tag = [token_ids[i][j]], cur_cls, 'B'
                else:
                    print(prev_tag)
                    raise KeyError

            elif l.startswith('I-'):
                if prev_tag == 'O':
                    prev_tag = 'I'
                    continue

                elif prev_tag == 'I':
                    cur_cls = int(l.split('L')[1])
                    if seq_cls == -1:
                        prev_tag = 'I'
                    elif cur_cls != seq_cls:
                        features[LABELNUM2NAME[seq_cls]].append(tokenizer.decode(accum_tokens))
                        accum_tokens, seq_cls, prev_tag = [], -1, 'I'
                    else:
                        accum_tokens.append(token_ids[i][j])
                        prev_tag = 'I'

                elif prev_tag == 'B':
                    cur_cls = int(l.split('L')[1])
                    if cur_cls != seq_cls:
                        features[LABELNUM2NAME[seq_cls]].append(tokenizer.decode(accum_tokens))
                        accum_tokens, seq_cls, prev_tag = [], -1, 'I'
                    else:
                        accum_tokens.append(token_ids[i][j])
                        prev_tag = 'I'

                else:
                    print(prev_tag)
                    raise KeyError

            elif l.startswith('O'):
                if prev_tag == 'O':
                    continue

                elif prev_tag == 'B':
                    features[LABELNUM2NAME[seq_cls]].append(tokenizer.decode(accum_tokens))
                    accum_tokens, seq_cls, prev_tag = [], -1, 'O'

                elif prev_tag == 'I':
                    if seq_cls == -1:
                        prev_tag == 'O'
                    else:
                        features[LABELNUM2NAME[seq_cls]].append(tokenizer.decode(accum_tokens))
                        accum_tokens, seq_cls, prev_tag = [], -1, 'O'

                else:
                    print(prev_tag)
                    raise KeyError

    if len(accum_tokens) > 0:
        features[LABELNUM2NAME[seq_cls]].append(tokenizer.decode(accum_tokens))

    accum_labels = reduce(lambda acc, v: acc + v, labels, [])
    return features, accum_labels