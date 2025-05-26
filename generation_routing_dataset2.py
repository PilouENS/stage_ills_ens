import os
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf-cache"
os.environ["HF_HOME"] = "/tmp/hf-home"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# === CONFIGURATION ===
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
max_seq_len = 128
num_samples = 10
top_k = 2
save_path = f"data/token_level_routing_wikitext_{num_samples}.pt"

os.makedirs("data", exist_ok=True)

# === CHARGEMENT DU DATASET ===
print("chargement du dataset ")
dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
texts = [t.strip() for t in dataset["text"] if len(t.strip()) > 0][:num_samples]

# === MODELE ET TOKENIZER ===
print("charg modèle et tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# === EXTRACTION TOKEN-LEVEL ===
token_level_data = []

print("extraction token par token...")
for prompt in tqdm(texts, desc="Processing prompts"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len).to(model.device)
    print("inputs", inputs)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=False,
            output_router_logits=True,
            use_cache=False,  
            return_dict=True
        )

    router_logits = outputs.router_logits  # list of tensors: (1, seq_len, num_experts)

    # On empile les top-2 experts pour chaque token de chaque couche
    selected_experts = []
    for layer_logits in router_logits:
        topk = torch.topk(layer_logits[0], k=top_k, dim=-1).indices  # (seq_len, k)
        selected_experts.append(topk.cpu())  # L x seq_len x k
    print("prompt", prompt)
    print("selected_experts", selected_experts)
    routing_path = torch.stack(selected_experts)  # (L, T, k)
    input_ids = inputs["input_ids"].squeeze(0).cpu()  # (T,)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    L, T, k = routing_path.shape
    for t in range(T):
        token_level_data.append({
            "token_str": tokens[t],
            "token_id": input_ids[t].item(),
            "routing_path": routing_path[:, t, :],  # (L, k)
        })

# === SAUVEGARDE ===
torch.save(token_level_data, save_path)
print(f" Dataset token-level sauvegardé dans {save_path} ({len(token_level_data)} tokens)")
