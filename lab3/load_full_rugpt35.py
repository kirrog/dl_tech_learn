from transformers import AutoTokenizer,AutoModelForCausalLM
from huggingface_hub import snapshot_download

model_name = "ai-forever/ruGPT-3.5-13B"
output_dir = '/home/ubuntu/DL/pretrained_models/ruGPT3_5'

print(model_name)
print(output_dir)

snapshot_download(repo_id=model_name, local_dir=output_dir, ignore_patterns=["LICENSE", "README.md", ".gitattributes"])
