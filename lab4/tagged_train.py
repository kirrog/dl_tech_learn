import time
import torch
from typing import List
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import get_scheduler
import gc
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from crf import CRF
from transformers import RobertaModel, AutoTokenizer
from datasets import load_metric
import json
import numpy as np
from dataclasses import dataclass
from typing import List
import evaluate
import torch.nn as nn
import torch

import warnings
warnings.filterwarnings("ignore")

#############################################

@dataclass
class Instance:
	words: List[str]
	ori_words: List[str]
	labels: List[str] = None
	prediction: List[str]  = None

#############################################

tagged_names = ["КТР","ЧСС","ТВП","ТРИКУСП РЕГУРГ","ХГЧ","PAPP-A","АФП",
                "Ингибин А","св эстрадиол","Возраст","Вес","Рост","ИМТ","ФК"]

LABEL_LIST = ['O']
for i, f_name in enumerate(tagged_names):
    f_b, f_i = f"B-L{i+1}", f"I-L{i+1}"
    LABEL_LIST.append(f_b)
    LABEL_LIST.append(f_i)
print(LABEL_LIST)

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

print(LABEL2ID)
print(ID2LABEL)

#############################################

ROOT_DIR = '/home/ubuntu/DL'

TRAIN_FILE = ROOT_DIR + '/medics/train.txt'
TEST_FILE = ROOT_DIR + '/medics/test.txt'
EVAL_FILE = ROOT_DIR + '/medics/dev.txt'
ROBERTA_PATH = ROOT_DIR + '/RuBioRoBERTa'

OUTPUT_DIR = ROOT_DIR + '/saved_model'
LOG_DIR = OUTPUT_DIR + '/logs'

BATCH_SIZE = 32
DEVICE = 'cuda'
EPOCHS = 30
LR = 2e-5
LAYERS_TO_HOLD = ['23', '22', '21', '20', '19', 'pooler']

print("BATCH_SIZE: ", BATCH_SIZE)
print("LR: ", LR)
print("LAYERS_TO_HOLD", LAYERS_TO_HOLD)

#############################################

print("Load base model")

class RobertaCRF(nn.Module):
    def __init__(self, label_size, roberta_path=ROBERTA_PATH, device=DEVICE):
        super(RobertaCRF, self).__init__()

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

TOKENIZER = AutoTokenizer.from_pretrained(ROBERTA_PATH, add_prefix_space=True, use_fast=True)
MODEL = RobertaCRF(len(LABEL_LIST)).to(DEVICE)
METRIC = evaluate.load("seqeval")
		
for name, param in MODEL.named_parameters():
	print(name, param.requires_grad)

#############################################

def compute_metrics(predictions, labels):

    # Remove ignored index (special tokens)
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label)]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label)]
        for prediction, label in zip(predictions, labels)
    ]

    results = METRIC.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

class NerDataset(Dataset):

    def __init__(self, file, label2idx, tokenizer):
        ## read all the instances. sentences and labels
        insts = self.read_file(file=file) 
        self.insts = insts
        self.label2idx = label2idx
        self.tokenizer = tokenizer

        len_input_ids = [
            len(self.tokenizer(self.insts[i].words, is_split_into_words=True)['input_ids']) 
            for i in range(len(self.insts))
            ]
        long_samples_ids = sorted(list(map(lambda vv: vv[0], filter(lambda v: v[1] > 512, enumerate(len_input_ids)))), reverse=True)
        for long_train_id in long_samples_ids:
            del self.insts[long_train_id]
        print(f"{len(long_samples_ids)} samples was deletet because its long len")

    def read_file(self, file: str, number: int = -1) -> List[Instance]:
        insts = []
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            ori_words = []
            labels = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    insts.append(Instance(words=words, ori_words=ori_words, labels=labels))
                    words = []
                    ori_words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                ls = line.split()
                word, label = ls[0],ls[-1]
                ori_words.append(word)
                words.append(word)
                labels.append(label)
        return insts

    def __len__(self):
        return len(self.insts)
    
    def tokenize_and_align_labels(self, example, label_all_tokens = True):
        tokenized_inputs = TOKENIZER(example.words, truncation=True, padding='max_length', max_length=512, 
                                     is_split_into_words=True)

        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(example.labels[word_idx])
            else:
                label_ids.append(example.labels[word_idx] if label_all_tokens else 0)
            previous_word_idx = word_idx

        tokenized_inputs["labels"] = list(map(lambda v: LABEL2ID.get(v, 0), label_ids))
        return tokenized_inputs

    def __getitem__(self, index):
        item = self.tokenize_and_align_labels(self.insts[index])
        return {'input_ids': torch.tensor(item['input_ids']), 
                'attention_mask': torch.ByteTensor(item['attention_mask']),
                'labels': torch.tensor(item['labels'])}
    
#############################################

print("Prepare Datasets")

train_ds = NerDataset(TRAIN_FILE,LABEL_LIST,TOKENIZER)
train_dataloader = DataLoader(train_ds, shuffle=True, 
                              batch_size=BATCH_SIZE)
print("train_ds: ", len(train_ds))

eval_ds = NerDataset(EVAL_FILE,LABEL_LIST,TOKENIZER)
val_dataloader = DataLoader(eval_ds, shuffle=False, 
                            batch_size=BATCH_SIZE)
print("eval_ds: ", len(eval_ds))

#############################################

OPTIMIZER = torch.optim.AdamW(MODEL.parameters(), lr=LR)

#############################################

print("START TRINING")

best_score = 0
for epoch in range(EPOCHS):
    print("Epoch: ", epoch)

    # Training stage
    MODEL.train()
    tr_loss = 0
    nb_tr_steps = 0
    train_tqdm = tqdm(train_dataloader)
    for batch in train_tqdm:
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        OPTIMIZER.zero_grad()
        loss = MODEL(input_ids=batch['input_ids'], 
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'], mode='train')
        
        loss.backward()
        OPTIMIZER.step()
        tr_loss += loss.item()
        nb_tr_steps += 1

        gc.collect()
        torch.cuda.empty_cache()

        show_dict = {'Train cur mean Loss': f'{tr_loss/nb_tr_steps:.6f}'}
        train_tqdm.set_postfix(show_dict)

    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    # Validation stage
    MODEL.eval()
    val_loss = 0
    nb_val_steps = 0
    true_labels,pred_labels = [],[]
    val_tqdm = tqdm(val_dataloader)
    for batch in val_tqdm:
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        with torch.no_grad():
            loss = MODEL(input_ids=batch['input_ids'], 
                         attention_mask=batch['attention_mask'],
                         labels=batch['labels'], mode='train')
            pred_labels = MODEL(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask'], mode='eval')

        val_loss += loss.item()
        nb_val_steps += 1

        gc.collect()
        torch.cuda.empty_cache()

        show_dict = {'Val cur mean Loss': f'{val_loss/nb_val_steps:.6f}'}
        val_tqdm.set_postfix(show_dict)

        predictions = pred_labels
        labels = batch["labels"].to('cpu').numpy().tolist()

        true_labels += labels
        pred_labels += predictions

    print("Validation loss: {}".format(val_loss/nb_val_steps))

    print("Compute metrics: ", end='')
    metrics = compute_metrics(pred_labels, true_labels)
    print("Ready!")
    print(metrics)

    # Save best model
    if metrics['f1'] > best_score:
        torch.save(MODEL.state_dict(), f"{OUTPUT_DIR}/ner_model")
        best_score = metrics['f1']

    print("Save epoch metrics: ", end='')
    metrics['val_loss'] = val_loss/nb_val_steps
    metrics['train_loss'] = tr_loss/nb_tr_steps
    metrics['epoch'] = epoch
    print(type(metrics))
    with open(f"{LOG_DIR}/epoch{epoch}", 'w', encoding='utf-8') as fd:
        fd.write(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("Ready!")