import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

name = '/home/ubuntu/DL/pretrained_models/ruGPT3_5'

# Loading the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    name,
    device_map='auto',
    load_in_8bit=True,
    max_memory={0: f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB'},
)
tokenizer = AutoTokenizer.from_pretrained(name)

SYSTEM_PROPMPT = (
    "Ты - поэт, который чётко следует инструкциям."
    "Твоя задача - написать одно стихотворение по заданному названию. "
    "Будь внимателен и не пиши ничего лишнего. "
    "Напиши только сам стих без указания дополниетльной информации к нему."
)

TITLE = 'Стихотворение о программисте'
PROMPT = f"###Система:\n{SYSTEM_PROPMPT}\n\n\n\n###Название:\n{TITLE}\n\n###Стихотворение:\n"

# Run text-generation pipeline
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Generate output
output = pipe(
    PROMPT,
    max_new_tokens=100,
    top_k=40,
    top_p=0.85,
    repetition_penalty=1.1,
    do_sample=True,
    use_cache=False,
)
print(output[0]['generated_text'])