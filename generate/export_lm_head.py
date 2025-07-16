import os
import torch

print("début du script ")

# Configuration de l'environnement
#redefinition des variables d'environnement pour Hugging Face
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf-datasets"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HOME"] = "/tmp/hf-home"

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    torch_dtype=torch.float16
)
model.eval()  # Met le modèle en mode évaluation
print(model.config)  # Affiche la configuration du modèle

with torch.no_grad():
    W_vocab = model.lm_head.weight             # vocab projection

torch.save(W_vocab, "W_vocab.pt")