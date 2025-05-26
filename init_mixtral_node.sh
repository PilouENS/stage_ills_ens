#!/bin/bash
# 1. On créé virtual env
cd ~/stage_ills_moe/stage_ILLS
python -m venv venv-mixtral
# 2. On l'active
source venv-mixtral/bin/activate

# 3. Installer pip s'il manque, et le mettre à jour
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools

# 4. Cloner le repo transformers s’il n’est pas déjà là
cd ~/stage_ills_moe/stage_ILLS
git clone https://github.com/huggingface/transformers.git

# 5. Se placer dans le dossier transformers
cd ~/stage_ills_moe/stage_ILLS/transformers

# 6. Installer transformers en mode editable (modifiable localement)
# 7. Installer torch et accelerate
pip install -e .
pip install torch
pip install accelerate
pip install scipy
pip install datasets
pip install matplotlib
pip install tqdm
# . Configuration des chemins de cache
export HF_HOME=/tmp/hf-home
export TRANSFORMERS_CACHE=/tmp/hf-cache
mkdir -p $HF_HOME $TRANSFORMERS_CACHE

# . Copier le token Hugging Face
if [ -f ~/.cache/huggingface/token ]; then
    echo "Copie du token Hugging Face vers /tmp..."
    cp ~/.cache/huggingface/token $HF_HOME/token
else
    echo "Token Hugging Face non trouvé"
    exit 1
fi



