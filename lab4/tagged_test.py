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
import torch
import evaluate
from transformers import RobertaModel, AutoTokenizer
from datasets import load_metric
import json
from crf import CRF
import numpy as np
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from functools import reduce

from dataclasses import dataclass
from typing import List

import warnings
warnings.filterwarnings("ignore")

#############################################

@dataclass
class Instance:
	words: List[str]
	ori_words: List[str]
	labels: List[str] = None
	prediction: List[str]  = None


##############################################

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

##############################################

ROOT_DIR = '.'

TEST_FILE = ROOT_DIR + '/medics/test.txt'
ROBERTA_PATH = ROOT_DIR + '/RuBioRoBERTa'
FINETUNED_MODEL_PATH = ROOT_DIR + '/saved_model/ner_model'


BATCH_SIZE = 4
DEVICE = 'cuda'
EPOCHS = 1
LR = 2e-5
LAYERS_TO_HOLD = ['23', '22', '21', 'pooler']

##############################################

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

MODEL.load_state_dict(torch.load(FINETUNED_MODEL_PATH))
MODEL.eval()

##############################################

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

    acc_f1 = 0
    for pred,label in zip(true_predictions, true_labels):
        acc_f1 += f1_score(pred,label, average='micro')
    
    print("f1 micro mean:", round(acc_f1 / len(true_predictions),3))

    results = METRIC.compute(predictions=true_predictions, references=true_labels)

    print(results)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
class NerDataset(Dataset):

    def __init__(self, file, label2idx, tokenizer):
        """
        sents: we use sentences if we want to build dataset from sentences directly instead of file
        """
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
            # Special tokens have a word id that is None. We set the label to 0 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(0)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(example.labels[word_idx])
            # For the other tokens in a word, we set the label to either the current label or 0, depending on
            # the label_all_tokens flag.
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
    
##############################################
    
test_ds = NerDataset(TEST_FILE,LABEL_LIST,TOKENIZER)
test_dataloader = DataLoader(test_ds, shuffle=False, batch_size=8)

##############################################

test_loss = 0
nb_test_steps = 0
true_labels,pred_labels = [],[]
test_tqdm = tqdm(test_dataloader)
for batch in test_tqdm:
    batch = {key: value.to(DEVICE) for key, value in batch.items()}
    with torch.no_grad():
        loss = MODEL(input_ids=batch['input_ids'], 
                         attention_mask=batch['attention_mask'],
                         labels=batch['labels'], mode='train')
        pred = MODEL(input_ids=batch['input_ids'], 
                                attention_mask=batch['attention_mask'], mode='eval')

    test_loss += loss.item()
    nb_test_steps += 1

    gc.collect()
    torch.cuda.empty_cache()

    show_dict = {'Test cur mean Loss': f'{test_loss/nb_test_steps:.6f}'}
    test_tqdm.set_postfix(show_dict)

    labels = batch["labels"].to('cpu').numpy().tolist()

    #print(predictions)
    #print(labels)

    true_labels += labels
    pred_labels += pred

print("Test loss: {}".format(test_loss/nb_test_steps))
print("pred/gold labels size: ", len(pred_labels), len(labels))


metrics = compute_metrics(pred_labels, true_labels)
print(metrics)
print(type(metrics))

# Save epoch metrics
metrics['true_labels'] = true_labels
metrics['pred_labels'] = pred_labels
metrics['test_loss'] = test_loss/nb_test_steps
with open(f"{ROOT_DIR}/test_metrics.json", 'w', encoding='utf-8') as fd:
    fd.write(json.dumps(metrics, indent=2, ensure_ascii=False))
