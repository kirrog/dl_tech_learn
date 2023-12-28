from peft import IA3Model, IA3Config, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import torch
from tqdm import tqdm
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import tqdm
import torch.nn as nn
from auto_gptq import AutoGPTQForCausalLM
import gc
from time import time

##############################

ROOT_DIR = '/home/ubuntu/DL/'

BASE_MODEL = ROOT_DIR + 'pretrained_models/fffrrt_ruGPT-3.5-13B-GPTQ4'
OUTPUT_DIR = ROOT_DIR
TRAIN_FILE = ROOT_DIR + 'data/train_part.csv'
EVAL_FILE = ROOT_DIR + 'data/test_part.csv'
LOG_DIR = OUTPUT_DIR + 'logs/'
ADAPTER_DIR = LOG_DIR + 'ai3_gpt3_5_4bit_adapter'

DEVICE = 'cuda:0'
BATCH_SIZE = 4
EPOCHS = 18
LR = 2e-4

print("BATCH_SIZE: ", BATCH_SIZE)
print("LR: ", LR)
print("EPOCHS: ", EPOCHS)
print("DEVICE: ", DEVICE)

##############################

print("Load tokenizer")
TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Init config")
CONFIG = IA3Config(
        peft_type="IA3",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj","c_fc"],
        feedforward_modules=['c_fc'])

print("Load base model")

s_time = time()
MODEL = AutoGPTQForCausalLM.from_quantized(BASE_MODEL, device=DEVICE, use_safetensors=True)
e_time = time()
print(f"Elapsed time: {round(e_time-s_time, 5)} sec.")

for param in MODEL.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp16 for stability
    param.data = param.data.to(torch.float16)

#MODEL.gradient_checkpointing_enable()  # reduce number of stored activations
#MODEL.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
MODEL.lm_head = CastOutputToFloat(MODEL.lm_head)

print("Wraps model to ia3")
IA3_MODEL = get_peft_model(MODEL, CONFIG)

#print(IA3_MODEL)

#for name, param in IA3_MODEL.named_parameters():
#    print(name, param.requires_grad)

# checking trainable parameters
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(IA3_MODEL)

OPTIMIZER = torch.optim.Adam(IA3_MODEL.parameters(), 
                             lr=LR)

##############################

def train_one(model, loader, optimizer):
    """Standard PyTorch training, one epoch"""
    model.train()
    acc_loss, counter = 0, 0
    process = tqdm.tqdm(loader)
    for batch in process:
        for k, v in batch.items():
            batch[k] = v.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'])

        loss = out['loss']
        
        loss.backward()
        OPTIMIZER.step()

        acc_loss += loss.item()
        counter += 1
        show_dict = {'Train cur mean Loss': f'{acc_loss/counter:.6f}'}
        process.set_postfix(show_dict)

        gc.collect()
        torch.cuda.empty_cache()

    return round(acc_loss/counter, 5)

def eval_one(model, loader):
    """Standard PyTorch evaluation, one epoch"""
    model.eval()
    acc_loss, counter = 0, 0
    process = tqdm.tqdm(loader)
    for batch in process:
        for k, v in batch.items():
            batch[k] = v.to(DEVICE)

        with torch.no_grad():
            out = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'])
            
        loss = out['loss']
        acc_loss += loss.item()
        counter += 1
        show_dict = {'Val cur mean Loss': f'{acc_loss/counter:.6f}'}
        process.set_postfix(show_dict)
        
        gc.collect()
        torch.cuda.empty_cache()

    return round(acc_loss/counter, 5)

class PoemsDataset(Dataset):
    def __init__(self, file, tokenizer):

        self.df = pd.read_csv(file, sep=';') 
        self.tokenizer = tokenizer
        self.text_col = 'text'
        self.title_col = 'title'
        self.end_of_text_token = '<|endoftext|>'

    def __len__(self):
        return self.df.shape[0]
    
    def formate_sample(self, index):
        text, title = self.df[self.text_col][index], self.df[self.title_col][index] 
        
        formated_string = f'Название: "{title}". Стихотворение: {text} {self.end_of_text_token}'        

        return formated_string        

    def __getitem__(self, index):
        return self.formate_sample(index)

def collate_fn(data):
    encoded_batch = TOKENIZER(data, return_tensors='pt', add_special_tokens=False, 
                              padding=True)

    return {
        'input_ids': encoded_batch['input_ids'],
        'attention_mask': encoded_batch['attention_mask'],
        'labels': encoded_batch['input_ids']
        }

##############################
print("Load dataset")

train_ds = PoemsDataset(TRAIN_FILE,TOKENIZER)
train_dataloader = DataLoader(train_ds, shuffle=True, 
                              batch_size=BATCH_SIZE, collate_fn=collate_fn)

eval_ds = PoemsDataset(EVAL_FILE,TOKENIZER)
eval_dataloader = DataLoader(eval_ds, shuffle=False, 
                             batch_size=BATCH_SIZE, collate_fn=collate_fn)

##############################
print("Start training")

gc.collect()
torch.cuda.empty_cache()

best_eval_loss, best_epoch = None, None
for i_epoch in range(EPOCHS):
     print(f"EPOCH: {i_epoch} | best_loss: {best_eval_loss} | best_epoch: {best_epoch}")
     loss_train = train_one(IA3_MODEL, train_dataloader, OPTIMIZER)
     loss_val = eval_one(IA3_MODEL, eval_dataloader)
     
     #
     print(f'{i_epoch} : loss_train={loss_train}, loss_val={loss_val}')
     metrics = {
          "epoch": i_epoch,
          "train_loss": loss_train, 
          "eval_loss": loss_val
          }
     
     # 
     if best_eval_loss is None or best_eval_loss >= loss_val:
          print("Saving new best AI3 adapter!")
          IA3_MODEL.save_pretrained(ADAPTER_DIR)
          best_eval_loss, best_epoch = loss_val, i_epoch
     elif i_epoch == (EPOCHS-1):
          print("Saving last AI3 adapter!")
          IA3_MODEL.save_pretrained(ADAPTER_DIR + '(last_train)')

     #
     m_object = json.dumps(metrics, indent=2, ensure_ascii=False)
     with open(LOG_DIR + f'epoch_{i_epoch}.json', 'w',encoding='utf-8') as fd:
          fd.write(m_object)

##############################