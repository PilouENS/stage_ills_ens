#!/bin/bash
cd $HOME/stage_ills_moe/stage_ILLS/pilou_git

# Initialise l’environnement
bash init_mixtral_node.sh

# Lance ton script Python
source $HOME/stage_ills_moe/stage_ILLS/venv-mixtral/bin/activate
cd $HOME/stage_ills_moe/stage_ILLS/pilou_git
python Generate_output.py
