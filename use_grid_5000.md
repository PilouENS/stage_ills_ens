# Environnement de Travail

## Grid5000
### Grid 5000 access
```
# user_name 
# mdp
ssh pfiloche@access.grid5000.fr
ssh rennes.g5k
```
### Réservation
#### Interactive
- `oarsub -I -l nodes=1/gpu=1,walltime=4:00:00`
- `oarsub -l host=1/gpu=2,walltime=0:30 -I`
- `oarsub -I -l ndodes=1/gpu=2,walltime=03:00:00 -q besteffort  -p "gpu_model='A40'" `  
- `oarsub -I -l gpu=1,walltime=03:00:00 -q besteffort -p "gpu_model='H100 NVL'"`
-I pour interactive, possible de spécifier nb de noeuds, de gpu/cpu, le modèle de gpu ou la mémoire totale des gpu. Pour Mixtral ~46GB je choisis souvent 2 A40.  
 `device_map="auto"` (arg de model) gère la répartition des tâches à travers les différents gpu dispo.
Possible de `breakpoint()` pour avoir une console interactive dans le script.

#### JOB (production)
Le script [submit_wait.sh](./pilou_git/submit_wait.sh) permet de lancer un job en production et de check son état.
- besteffort : job commence vite mais peut être interrompu
- production :  ça met mille ans à avoir un noeud mais pas d'interruptions (regarder comment passer de p3 à p1)
Script à modifier suivant les ressources voulues et le walltime (temps max du job). Il lance le script `run_on_node.sh` dans lequel on met en place notre environnement de travail avec `init_mixtral_node.sh` et on lance le script de génération. 

## Outils pratiques 
Visualisation des gpu :
lspci | grep -i nvidia

### Surcharge du $home
Pour ne pas surcharger le home en quantité de données on import les modèles et dataset HuggingFace sur le tmp du noeud. 

### [init_mixtral_node.sh](./pilou_git/init_mixtral_node.sh)
A FAIRE SUR LE NOEUD
Script bash pour mettre en place l'environnement :
- création d'un venv
- maj de pip et install des bibli py
- gestion token huggin face 
- on se place sur le noeud pour telecharger
- download transformers en editable 





python-grid5000 : https://gitlab.inria.fr/msimonin/python-grid5000/-/tree/master?ref_type=heads