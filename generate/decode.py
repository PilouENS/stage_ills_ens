import torch
import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HOME"] = "/tmp/hf-home"

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

token_to_id = tokenizer.get_vocab()  # dict: token → id
id_to_token = {id_: tok for tok, id_ in token_to_id.items()} # dict: id → token

torch.save(token_to_id, "token_to_id.pt")  # Save the vocabulary to a file
torch.save(id_to_token, "id_to_token.pt")  # Save the id_to_token mapping to a file