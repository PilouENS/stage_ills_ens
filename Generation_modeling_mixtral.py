"""
script pour générer des sorties à partir du modèle Mixtral-8x7B-Instruct-v0.1
Il faut modifier modeling_mixtral.py pour avoir les sorties 
fichiers modeling_mixtral_modifée dispo dans pilou_git/modeling_mixtral/
"""
import os
import torch
from datasets import load_dataset
from tqdm import tqdm

# Configuration de l'environnement
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HOME"] = "/tmp/hf-home"

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.mixtral.modeling_mixtral import resultat  # Uncomment and adjust if 'resultat' is needed and available in your PYTHONPATH

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained
            (
                model_id, 
                device_map="auto", 
                torch_dtype=torch.float16,
            )
model.eval()  # Met le modèle en mode évaluation

### === Chargement du dataset === ###   
#dataset = load_dataset("HuggingFaceH4/helpful_instructions", split="train[:1000]")
#dataset = load_dataset("HuggingFaceH4/testing_codealpaca_small", split="train[:1000]", trust_remote_code=True)
dataset = load_dataset('flytech/python-codes-25k', split="train[:1000]", trust_remote_code=True)
print(dataset[0].keys())

for sample in tqdm(dataset):  # progress bar utile pour les longs datasets
    #prompt = sample["prompt"] #helpful_instructions
    prompt = sample.get("prompt", "").strip()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)

#on enregistre les resultats dans un fcihier
torch.save(resultat, "router_logits_codealpaca.pt")

#pour un prompt on genere une reponse   
prompt = dataset[0].get("prompt", "").strip()
print("Prompt :", prompt)
input = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(input.input_ids, max_new_tokens=50, return_dict_in_generate=True, output_router_logits=True)
print("Réponse générée :", tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))




"""
print(inputs["input_ids"].shape)
print("logits de la première couche", router_logits[0])  # logits de la première couche
print("logits de la dernière couche", router_logits[1])  # logits de la dernière couche

# Affichage des top-2 experts par token pour chaque couche
for layer_idx, logits in enumerate(router_logits):
    top2 = torch.topk(logits, k=2, dim=-1).indices
    #print(f"Layer {layer_idx} → top-2 experts par token :", top2)

"""