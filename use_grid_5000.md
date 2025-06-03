# Environnement de Travail

## Grid5000
### Grid 5000 access
`
user_name 
mdp
ssh pfiloche@access.grid5000.fr
ssh rennes.g5k
`
### Réservation
#### Interactive
- `oarsub -I -l nodes=1/gpu=1,walltime=4:00:00`
- `oarsub -l host=1/gpu=2,walltime=0:30 -I`
- `oarsub -I -l ndodes=1/gpu=2,walltime=03:00:00 -q besteffort  -p "gpu_model='A40'" `

#### JOB (production)
Le script [submit_wait.sh](./pilou_git/submit_wait.sh) permet de lancer un job en production et de check son état.
- besteffort : job commence vite mais peut être interrompu
- production :  ça met mille ans à avoir un noeud mais pas d'interruptions (regarder comment passer de p3 à p1)
Il lance 

Visualisation des gpu :
lspci | grep -i nvidia
## [init_mixtral_node.sh](./pilou_git/init_mixtral_node.sh)
A FAIRE SUR LE NOEUD
Script bash pour mettre en place l'environnement :
- création d'un venv
- maj de pip et install des bibli py
- gestion token huggin face 
- on se place sur le noeud pour telecharger
- download transformers en editable 
à détailler 
Pour lancer un job sans interactive mode : submit_wait.sh




reservation interactive :
oarsub -I -l nodes=1/gpu=1,walltime=4:00:00
oarsub -l host=1/gpu=2,walltime=0:30 -I
oarsub -I -l ndodes=1/gpu=2,walltime=03:00:00 -q besteffort -p "gpu_model='A40'"
(il faut check sur le site pour d'autres modèles)
Visualisation des gpu :
lspci | grep -i nvidia

utilisation des gpu :
nvidia-smi


enter ipython :
grid5000

python-grid5000 : https://gitlab.inria.fr/msimonin/python-grid5000/-/tree/master?ref_type=heads