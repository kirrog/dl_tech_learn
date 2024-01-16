from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, prepare_model_for_int8_training

###########################

model_path = "/home/ubuntu/DL/pretrained_models/ruGPT3_5"
adapter_path = "/home/ubuntu/DL/logs/ai3_gpt3_5_8bit_adapter"
DEVICE = 'cuda:0'
SYSTEM_PROPMPT = (
    "Ты - поэт, который чётко следует инструкциям."
    "Твоя задача - написать одно стихотворение в стиле указанного автора по заданному названию. "
    "Будь внимателен и не пиши ничего лишнего. "
    "Напиши только сам стих без указания дополниетльной информации к нему."
)
MAX_SEQ_LEN = 400
PARAMS = {
    'do_sample': True,
    'top_k':10, 'top_p': 0.95,
    'repetition_penalty': 0.9,
    'temperature': 14.4,
    'no_repeat_ngram_size': 0 
}

MODEL = AutoModelForCausalLM.from_pretrained(model_path, 
                                             device_map=DEVICE, load_in_8bit=True,
                                             torch_dtype=torch.float16,use_cache=True)
TOKENIZER = AutoTokenizer.from_pretrained(model_path)

MODEL = prepare_model_for_int8_training(MODEL)
MODEL = PeftModel.from_pretrained(MODEL, adapter_path)

###########################

def generate_poem(title, author):
    prompt = f"###Система:\n{SYSTEM_PROPMPT}\n\n\n\n###Автор:\n{author}\n\n###Название:\n{title}\n\n###Стихотворение:\n"
    encoded_prompt = TOKENIZER(prompt, return_tensors='pt', add_special_tokens=False)

    generated_ids = MODEL.generate(max_length=MAX_SEQ_LEN , num_return_sequences=1, 
                                   eos_token_id=TOKENIZER.eos_token_id, pad_token_id=TOKENIZER.eos_token_id,
                                                 **encoded_prompt,**PARAMS)
    
    gen_txt = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)

    return gen_txt[0]

###########################

EXAMPLE_INPUTS = [
    ("Погода", "Александр Пушкин"),
    ("Погода", "Владимир Высоцкий"),
    ("Погода", "Фёдор Достоевский"),
    ("Погода", "Анна Ахматова")
]

for i, item in enumerate(EXAMPLE_INPUTS):
    print(f"СТИХ {i}:")
    print(generate_poem(item[0], item[1]))
    print()