import optuna
import joblib
import evaluate
from time import time
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import gc
import torch

########################################################

ROOT_DIR = './'
OUTPUT_DIR = ROOT_DIR + 'gen_search/'
METRIC = evaluate.load("chrf")
MODEL_PATH = ''
DEVICE = 'cuda:0'
MAX_SEQ_LEN = 600
REFS_FILE = ROOT_DIR + 'data/test_part.csv'

########################################################

class SearchUtils:
    #
    search_config = {
        'beamsearch': lambda trial: {
            'num_beams': trial.suggest_categorical("num_beams", [8,10,12,14,16,18,20]),
            'num_beam_groups': trial.suggest_categorical('num_beam_groups', [2,4]),
            'do_sample': trial.suggest_categorical('do_sample', [True, False]),
            'early_stopping': trial.suggest_categorical('early_stopping', [True, False])},

        'sampling': lambda trial: {
            'do_sample': True,
            'top_k': trial.suggest_int('top_k', 0, 51, 5),
            'top_p': trial.suggest_float('top_p',0.5,0.96,0.05)},

        'contrastivesearch': lambda trial: {
            'top_k': trial.suggest_int('top_k', 2, 50, 2),
            'penalty_alpha': trial.suggest_float('penalty_alpha',0.5,0.9,0.05)
        }
    }

    #
    def __init__(self, model_path, refs_file, max_length) -> None:
        self.init_model(model_path)
        self.formate_refs(refs_file)
        self.generate_type = None
        self.max_gen_seq_len = max_length
        self.trial_counter = None

    #
    def formate_refs(self, refs_file):
        df = pd.read_csv(refs_file, sep=';').loc[:800,:].reset_index(drop=True)

        #
        self.prompt_texts = [f'Название: "{df["title"][i]}". Стихотворение:' for i in range(df.shape[0])]
        #
        self.ref_texts = [[f"{prompt_t} {self.df['text'][i]}"] for i, prompt_t in enumerate(self.prompt_texts)]
        #
        s_time = time()
        self.enc_prompts = [self.tokenizer(prompt_t, return_tensors='pt', add_special_tokens=False) 
                            for prompt_t in self.prompt_texts]
        e_time = time()
        print(f"Elapsed time: {round(e_time-s_time, 5)} sec.")

    # 
    def init_model(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        s_time = time()
        self.model = AutoGPTQForCausalLM.from_quantized(model_path, device=DEVICE, 
                                                        use_safetensors=True)
        e_time = time()
        print(f"Elapsed time: {round(e_time-s_time, 5)} sec.")

    #
    def compute_metrics(self, refs, preds):
        return METRIC(predictions=preds, 
                      references=refs, word_order=2)['score']

    #
    def generate_samples(self, enc_prompts, params):
        generated_texts = []
        for i, enc_prompt in tqdm(enumerate(enc_prompts)):
            enc_prompt = {k: v.to(DEVICE) for k,v in enc_prompt.items()}

            gen_encode = self.model.generate(max_length=self.max_gen_seq_len, 
                                             num_return_sequences=1, eos_token_id=self.model.config.eos_token_id,
                                             **enc_prompt,**params)
            
            gen_txt = self.tokenizer.batch_decode(gen_encode, skip_special_tokens=True)
            
            print(f"== [trial - {self.trial_counter} | sample - {i}] GENERATED TEXT START: ")
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
        selected_params = self.search_config[self.generate_type](trials)
        #
        pred_texts = self.generate_samples(self.enc_prompts, 
                                           selected_params)
        #
        charf_score = self.compute_metrics(self.ref_texts, 
                                           pred_texts)

        print(f"\n== TRIAL FINAL CHARF++ SCORE: {charf_score} ==\n")

        return charf_score
    
    #
    def save_search(self, study, study_name):
        joblib.dump(study, f'{OUTPUT_DIR}{study_name}_study.pkl')

#
def search(study_name, utils, n_trials):
    print(f"====== START SEARCH ({study_name}) ======")

    study = optuna.create_study(directions=["maximize"])
    utils.generate_type = study_name
    utils.trial_counter = 0
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
utils = SearchUtils(MODEL_PATH, REFS_FILE, MAX_SEQ_LEN)

########################################################

print("=== START SEARCHING... ===")

#
search("beamsearch", utils, 28)

#
search("sampling", utils, 40)

#
search("contrastivesearch", utils, 40)

########################################################

print("=== DONE! ===")