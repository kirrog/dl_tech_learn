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
import bitsandbytes as bnb
from transformers import get_linear_schedule_with_warmup
from math import ceil
from peft import prepare_model_for_int8_training
import random

np.random.seed(10)
random.seed(10)

##############################

ROOT_DIR = '/home/ubuntu/DL/'

SYSTEM_PROPMPT = (
    "Ты - поэт, который чётко следует инструкциям."
    "Твоя задача - написать одно стихотворение в стиле указанного автора по заданному названию. "
    "Будь внимателен и не пиши ничего лишнего. "
    "Напиши только сам стих без указания дополниетльной информации к нему."
)

BASE_MODEL = ROOT_DIR + 'pretrained_models/ruGPT3_5'
OUTPUT_DIR = ROOT_DIR
TRAIN_FILE = ROOT_DIR + 'data/train_part.csv'
EVAL_FILE = ROOT_DIR + 'data/test_part.csv'
LOG_DIR = OUTPUT_DIR + 'logs/'
ADAPTER_DIR = LOG_DIR + 'ai3_gpt3_5_8bit_adapter'

DEVICE = 'cuda:0'
NUM_ACCUMULATION_STEPS = 16
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
WARMUP_STEPS = 2

EPOCHS = 10
LR = 2e-4

print("SYTEM_PROMPT: ", SYSTEM_PROPMPT)
print("NUM_ACCUMULATION_STEPS: ", NUM_ACCUMULATION_STEPS)
print("WARMUP_STEPS: ", WARMUP_STEPS)
print("TRAIN_BATCH_SIZE: ", TRAIN_BATCH_SIZE)
print("EVAL_BATCH_SIZE: ", EVAL_BATCH_SIZE)
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
MODEL = AutoModelForCausalLM.from_pretrained(BASE_MODEL, 
                                             device_map=DEVICE, load_in_8bit=True,
                                             torch_dtype=torch.float16,
                                             use_cache=False)
e_time = time()
print(f"Elapsed time: {round(e_time-s_time, 5)} sec.")

print("Prepare model for 8bit tuning")
for param in MODEL.parameters():
    param.requires_grad = False
MODEL = prepare_model_for_int8_training(MODEL)

print("Wraps model to ia3")
IA3_MODEL = get_peft_model(MODEL, CONFIG)

print("Model architecture:")
print(IA3_MODEL)

print("Model layers info:")
for name, param in IA3_MODEL.named_parameters():
    print(name, param.requires_grad, param.data.dtype)

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

#OPTIMIZER = torch.optim.Adam(IA3_MODEL.parameters(),lr=LR)
OPTIMIZER = bnb.optim.Adam8bit(IA3_MODEL.parameters(), lr=LR)

##############################

def train_one(model, loader, optimizer, scheduler):
    """Standard PyTorch training, one epoch"""
    model.train()
    acc_loss, step, steps_amount = 0, 0, (len(loader) // NUM_ACCUMULATION_STEPS)
    acc_losses_lst, sum_acc_losses, lrs = [], 0, []

    process = tqdm.tqdm(loader)
    for idx, batch in enumerate(process):
        process.set_postfix({})
        for k, v in batch.items():
            batch[k] = v.to(DEVICE)
        
        out = model(input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'])

        # Normalize the Gradients
        loss = out['loss']
        norm_loss = loss / NUM_ACCUMULATION_STEPS
        norm_loss.backward()
        acc_loss += norm_loss.item()

        # Update Optimizer and Scheduler
        if ((idx+1) % NUM_ACCUMULATION_STEPS == 0) or ((idx+1) == len(loader)):
            optimizer.step()
            lrs.append(round(optimizer.param_groups[0]["lr"],8))

            scheduler.step()
            optimizer.zero_grad()

            acc_losses_lst.append(acc_loss)
            sum_acc_losses += acc_loss
            step += 1

            # Displat train info
            show_dict = {
                'Step': f"{step} / {steps_amount}", 
                'Cur mean acc_loss': f'{sum_acc_losses/step:.6f}', 
                'Cur accumulated loss': f"{acc_loss:.6f}",
                'Cur learning_rate': f"{lrs[-1]}"}
            process.set_postfix(show_dict)
            acc_loss = 0

        gc.collect()
        torch.cuda.empty_cache()

    optimizer.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()
    return round(sum_acc_losses/step, 6), acc_losses_lst, lrs

def eval_one(model, loader):
    """Standard PyTorch evaluation, one epoch"""
    model.eval()
    acc_loss = 0

    process = tqdm.tqdm(loader)
    for idx, batch in enumerate(process):
        for k, v in batch.items():
            batch[k] = v.to(DEVICE)

        with torch.no_grad():
            out = model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'])
            
        loss = out['loss']
        acc_loss += loss.item()
        show_dict = {'Val cur mean Loss': f'{acc_loss/(idx+1):.6f}'}
        process.set_postfix(show_dict)
        
        gc.collect()
        torch.cuda.empty_cache()

    return round(acc_loss/len(loader), 5)

class PoemsDataset(Dataset):
    def __init__(self, file, tokenizer):

        self.df = pd.read_csv(file, sep=';') 
        self.tokenizer = tokenizer
        self.text_col = 'text'
        self.title_col = 'title'
        self.author_col = 'author'
        self.end_of_text_token = '<|endoftext|>'

    def __len__(self):
        return self.df.shape[0]
    
    def formate_sample(self, index):
        text, title, author = self.df[self.text_col][index], self.df[self.title_col][index], self.df[self.author_col][index] 
        
        formated_string = f"###Система:\n{SYSTEM_PROPMPT}\n\n\n\n###Автор:\n{author}\n\n###Название:\n{title}\n\n###Стихотворение:\n{text}\n\n{self.end_of_text_token}"

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
train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=TRAIN_BATCH_SIZE, 
                              collate_fn=collate_fn, drop_last=True, worker_init_fn=np.random.seed(10))

eval_ds = PoemsDataset(EVAL_FILE,TOKENIZER)
eval_dataloader = DataLoader(eval_ds, shuffle=False, 
                             batch_size=EVAL_BATCH_SIZE, collate_fn=collate_fn)

##############################

t_total = len(train_dataloader) // NUM_ACCUMULATION_STEPS * EPOCHS
SCHEDULER = get_linear_schedule_with_warmup(
        OPTIMIZER, num_warmup_steps=WARMUP_STEPS, num_training_steps=t_total)


#############################

print("Start training")

gc.collect()
torch.cuda.empty_cache()

best_eval_loss, best_epoch = None, None
for i_epoch in range(EPOCHS):
     print(f"EPOCH: {i_epoch} | best_loss: {best_eval_loss} | best_epoch: {best_epoch}")
     loss_train, losses_list, lrs = train_one(IA3_MODEL, train_dataloader, OPTIMIZER, SCHEDULER)
     loss_val = eval_one(IA3_MODEL, eval_dataloader)
     
     #
     print(f'{i_epoch} : loss_train={loss_train}, loss_val={loss_val}')
     metrics = {
          "epoch": i_epoch,
          "train_loss": loss_train,
          "train_losses_list": losses_list,
          "train_learning_rates": lrs,
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