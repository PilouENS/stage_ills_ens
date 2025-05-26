import os
import torch
from datasets import load_dataset

# Configuration de l'environnement
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HOME"] = "/tmp/hf-home"

from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
model.config.output_router_logits = True  # pour voir les sorties du routeur
dataset = load_dataset("HuggingFaceH4/helpful_instructions", split="train[:1000]")  # ← juste 10 exemples pour tester


#print(dataset[0].keys())
instructions = [sample["prompt"] for sample in dataset]
gros_texte = "\n\n".join(instructions)
inputs = tokenizer(gros_texte, return_tensors="pt", truncation=False).to(model.device)
#print("gros_texte", gros_texte)
with torch.no_grad():
    outputs = model(**inputs, return_dict=True)

    #router_logits = outputs.router_logits  # Liste : une entrée par couche

"""
print(inputs["input_ids"].shape)
print("logits de la première couche", router_logits[0])  # logits de la première couche
print("logits de la dernière couche", router_logits[1])  # logits de la dernière couche

# Affichage des top-2 experts par token pour chaque couche
for layer_idx, logits in enumerate(router_logits):
    top2 = torch.topk(logits, k=2, dim=-1).indices
    #print(f"Layer {layer_idx} → top-2 experts par token :", top2)

"""