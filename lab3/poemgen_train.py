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

##############################

BASE_MODEL = '/home/ubuntu/DL/pretrained_models/ruGPT3_5_13B_8bit'
OUTPUT_DIR = '/home/ubuntu/DL/'
TRAIN_FILE = '/home/ubuntu/DL/data/train_part.csv'
EVAL_FILE = '/home/ubuntu/DL/data/test_part.csv'
LOG_DIR = OUTPUT_DIR + 'logs/'
ADAPTER_DIR = LOG_DIR + 'ai3_gpt3_5_8bit_adapter'

DEVICE = 'cuda:0'
BATCH_SIZE = 1
EPOCHS = 12
LR = 3e-4

print("BATCH_SIZE: ", BATCH_SIZE)
print("LR: ", LR)
print("EPOCHS: ", EPOCHS)
print("DEVICE: ", DEVICE)

##############################

from auto_gptq import AutoGPTQForCausalLM

print("Load tokenizer")
TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL)

print("Init config")
CONFIG = IA3Config(
        peft_type="IA3",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj","c_fc"],
        feedforward_modules=['c_fc'])

print("Load base model")
#MODEL = AutoModelForCausalLM.from_pretrained(BASE_MODEL, #load_in_8bit=True, 
#                                             use_safetensors=True, device_map={'': 0})
MODEL = AutoGPTQForCausalLM.from_quantized(BASE_MODEL, device=DEVICE)

# frozing full base model
for param in MODEL.parameters():
    param.requires_grad = False

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


input_ids = torch.tensor(np.random.randint(30, 50, size=(1,1,94)))
print(MODEL(input_ids=input_ids))

##############################

def train_one(model, loader, optimizer):
    """Standard PyTorch training, one epoch"""
    model.train()
    acc_loss, counter = [], 1
    process = tqdm.tqdm(loader)
    for batch in process:
        print(batch['input_ids'].size(), batch['attention_mask'].size(), batch['labels'].size())

        for k, v in batch.items():
            batch[k] = v.to(DEVICE)

        print(batch['input_ids'].size(), batch['attention_mask'].size(), batch['labels'].size())
        
        optimizer.zero_grad()
        out = model(input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'])
        
        loss = out['loss']
        acc_loss += loss.item()
        show_dict = {'Train cur mean Loss': f'{acc_loss/counter:.6f}'}
        process.set_postfix(show_dict)

    return round(acc_loss/counter, 5)

def eval_one(model, loader):
    """Standard PyTorch evaluation, one epoch"""
    model.eval()
    acc_loss, counter = [], 1
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
        show_dict = {'Val cur mean Loss': f'{acc_loss/counter:.6f}'}
        process.set_postfix(show_dict)

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
        
        formated_sttring = f'Название: "{title}". Стихотворение: {text} {self.end_of_text_token}'        
        encoded_string = self.tokenizer(formated_sttring, return_tensors='pt', add_special_tokens=True)

        return {
            'input_ids': encoded_string['input_ids'],
            'attention_mask': encoded_string['attention_mask'],
            'labels': encoded_string['input_ids']
        }

    def __getitem__(self, index):
        return self.formate_sample(index)

##############################
print("Load dataset")

train_ds = PoemsDataset(TRAIN_FILE,TOKENIZER)
train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=1)

eval_ds = PoemsDataset(EVAL_FILE,TOKENIZER)
eval_dataloader = DataLoader(eval_ds, shuffle=False, batch_size=1)

##############################
print("Start training")

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
     with open(LOG_DIR + f'epoch_{i_epoch}', 'w',encoding='utf-8') as fd:
          fd.write(m_object)

##############################