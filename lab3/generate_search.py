import optuna
import joblib
import evaluate
from time import time
from auto_gptq import AutoGPTQForCausalLM
from peft import PeftModel 
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import gc
import torch
import torch.nn as nn
import random 
import os
from peft import prepare_model_for_int8_training
from transformers import AutoTokenizer, AutoModelForCausalLM

random.seed(10)

import logging
logging.basicConfig(level = logging.INFO)

os.environ["PYTHONUNBUFFERED"] = "true"

########################################################

print("=== SET HYPERP ===")

SYSTEM_PROPMPT = (
    "Ты - поэт, который чётко следует инструкциям."
    "Твоя задача - написать одно стихотворение в стиле указанного автора по заданному названию. "
    "Будь внимателен и не пиши ничего лишнего. "
    "Напиши только сам стих без указания дополниетльной информации к нему."
)

ROOT_DIR = '/home/ubuntu/DL/'
OUTPUT_DIR = ROOT_DIR + 'gen_search/'
METRIC = evaluate.load("chrf")
MODEL_PATH = ROOT_DIR + 'pretrained_models/ruGPT3_5'
ADAPTER_PATH = ROOT_DIR + 'logs/ai3_gpt3_5_8bit_adapter'
DEVICE = 'cuda:0'
MAX_SEQ_LEN = 600
REFS_FILE = ROOT_DIR + 'data/test_part.csv'
SAMPLE_AMOUNT = 10
REF_TEXT_LEN_LIMIT = 1000

########################################################

class SearchUtils:
    #
    search_config = {
        'beamsearch': lambda trial: {
            'num_beams': trial.suggest_categorical("num_beams", [8,10,12,14,16,18,20]),
            'num_beam_groups': trial.suggest_categorical('num_beam_groups', [2,4]),
            'early_stopping': True,
            'diversity_penalty': trial.suggest_float('diversity_penalty', 0.1, 10, step=0.1),
            'repetition_penalty': trial.suggest_float('repetition_penalty', 0.0, 10, step=0.1),
            'no_repeat_ngram_size': trial.suggest_categorical('no_repeat_ngram_size', [0,1,2,3,4])
        },

        'sampling': lambda trial: {
            'do_sample': True,
            'top_k': trial.suggest_int('top_k', 0, 51, step=5),
            'top_p': trial.suggest_float('top_p',0.5, 0.96, step=0.05),
            'early_stopping': True,
            'repetition_penalty': trial.suggest_float('repetition_penalty', 0.0, 10, step=0.1),
            'temperature': trial.suggest_float('temperature', 0.0, 15.0, step=0.2),
            'no_repeat_ngram_size': trial.suggest_categorical('no_repeat_ngram_size', [0,1,2,3,4])
        },

        'contrastivesearch': lambda trial: {
            'top_k': trial.suggest_int('top_k', 2, 50, step=2),
            'penalty_alpha': trial.suggest_float('penalty_alpha', 0.5, 0.9, step=0.05),
            'early_stopping': True,
            'repetition_penalty': trial.suggest_float('repetition_penalty', 0.0, 10, step=0.1),
            'temperature': trial.suggest_float('temperature', 0.0, 15.0, step=0.2),
            'no_repeat_ngram_size': trial.suggest_categorical('no_repeat_ngram_size', [0,1,2,3,4])
        }
    }

    #
    def __init__(self, model_path, adapter_path, refs_file, max_length) -> None:
        self.init_model(model_path, adapter_path)
        self.formate_refs(refs_file)
        self.generate_type = None
        self.max_gen_seq_len = max_length
        self.trial_counter = None

    #
    def formate_refs(self, refs_file):
        
        df = pd.read_csv(refs_file, sep=';')
        df = df[df['text'].str.len() <= REF_TEXT_LEN_LIMIT]
        selected_ids = random.choices(list(df.index), k=SAMPLE_AMOUNT)

        print("== SELECTED REF IDS: ")
        print(selected_ids)

        df = df.loc[selected_ids,:].reset_index(drop=True)

        #
        self.prompt_texts = [f"###Система:\n{SYSTEM_PROPMPT}\n\n\n\n###Автор:\n{df['author'][i]}\n\n###Название:\n{df['title'][i]}\n\n###Стихотворение:\n" for i in range(df.shape[0])]

        #
        self.ref_texts = [[f"{prompt_t}{df['text'][i]}\n\n"] for i, prompt_t in enumerate(self.prompt_texts)]
        
        #
        s_time = time()
        self.enc_prompts = [self.tokenizer(prompt_t, return_tensors='pt', add_special_tokens=False) 
                            for prompt_t in self.prompt_texts]
        e_time = time()
        print(f"Elapsed time: {round(e_time-s_time, 5)} sec.")

        print(len(self.prompt_texts), len(self.ref_texts), len(self.enc_prompts))

    # 
    def init_model(self, model_path, adapter_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        s_time = time()
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          device_map=DEVICE, load_in_8bit=True,
                                                          torch_dtype=torch.float16,use_cache=False)
        self.model = prepare_model_for_int8_training(self.model)
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        e_time = time()
        print(f"Elapsed time: {round(e_time-s_time, 5)} sec.")

        #
        self.model.eval()
        gc.collect()
        torch.cuda.empty_cache()

    #
    def compute_metrics(self, refs, preds):
        return METRIC.compute(predictions=preds, 
                      references=refs, word_order=2)['score']

    #
    def generate_samples(self, enc_prompts, params):
        generated_texts = []
        for i, enc_prompt in enumerate(enc_prompts):
            print(f"== [trial - {self.trial_counter} | sample - {i} / {SAMPLE_AMOUNT}]")

            gc.collect()
            torch.cuda.empty_cache()

            for k, v in enc_prompt.items():
                enc_prompt[k] = v.to(DEVICE)

            print(enc_prompt)

            print("== GENERATING...", end='')
            s_time = time()
            with torch.no_grad():
                gen_encode = self.model.generate(max_length=self.max_gen_seq_len, 
                                                 num_return_sequences=1, 
                                                 eos_token_id=self.tokenizer.eos_token_id,
                                                 pad_token_id=self.tokenizer.eos_token_id,
                                                 **enc_prompt,**params)
            e_time = time()
            print(f"[{round(e_time-s_time, 3)} sec.]")

            print("== DECODING...", end='')
            s_time = time()
            gen_txt = self.tokenizer.batch_decode(gen_encode, skip_special_tokens=True)
            e_time = time()
            print(f"[{round(e_time-s_time, 3)} sec.]")
            
            print(f"== [{i}] GENERATED TEXT START ==")
            print(gen_txt)
            print(f"== [{i}] GENERATED TEXT END ==")
            
            generated_texts += gen_txt

            gc.collect()
            torch.cuda.empty_cache()

        return generated_texts
    
    #
    def search_objective(self, trials):
        self.trial_counter += 1

        #
        print("== SELECTING SAMPLES... ")
        selected_params = self.search_config[self.generate_type](trials)
        
        #
        print("== GENERATING SAMPLES...")
        try:
            pred_texts = self.generate_samples(self.enc_prompts, 
                                               selected_params)
            
        # invalid generation parameters values
        except ValueError:
            return 0
        #
        print("== COMPUTING METRIC...")
        chrf_score = self.compute_metrics(self.ref_texts, 
                                           pred_texts)

        print(f"\n== TRIAL FINAL CHRF++ SCORE: {chrf_score} ==\n")

        return chrf_score
    
    #
    def save_search(self, study, study_name):
        joblib.dump(study, f'{OUTPUT_DIR}{study_name}_study.pkl')

#
def search(study_name, utils, n_trials):
    print(f"====== START SEARCH ({study_name}) ======")

    study = optuna.create_study(directions=["maximize"])
    utils.generate_type = study_name
    utils.trial_counter = 0
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(utils.search_objective, n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))

    print(f"\n== TRIALS INFO:")
    print(study.trials)
    print("==\n")

    print(f"====== END SEARCH ({study_name}) ======")

    utils.save_search(study, study_name)

    print(f"====== SEARCH SAVED ======")

########################################################

print("=== LOAD UTILS CLASS ===")
utils = SearchUtils(MODEL_PATH, ADAPTER_PATH, REFS_FILE, MAX_SEQ_LEN)

########################################################

print("=== START SEARCHING... ===")

#
#BM_NAME, BM_TRIALS_AMOUNT = "beamsearch", 15
#print(f"== {BM_NAME}: {BM_TRIALS_AMOUNT} trials amount" )
#search(BM_NAME, utils, BM_TRIALS_AMOUNT)

#
S_NAME, S_TRIALS_AMOUNT = "sampling", 15
print(f"== {S_NAME}: {S_TRIALS_AMOUNT} trials amount")
search(S_NAME, utils, S_TRIALS_AMOUNT)

#
CS_NAME, CS_TRIALS_AMOUNT = "contrastivesearch", 15
print(f"== {CS_NAME}: {CS_TRIALS_AMOUNT} trials amount")
search(CS_NAME, utils, CS_TRIALS_AMOUNT)

########################################################

print("=== DONE! ===")