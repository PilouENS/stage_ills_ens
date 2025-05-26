Grid 5000 access
user_name
mdp

ssh pfiloche@access.grid5000.fr
ssh rennes.g5k

reservation interactive :
oarsub -I -l nodes=1/gpu=1,walltime=4:00:00
oarsub -l host=1/gpu=2,walltime=0:30 -I
oarsub -I -l gpu=1,walltime=03:00:00 -q besteffort -p "gpu_model='H100 NVL'"
(il faut check sur le site pour d'autres modèles)
Visualisation des gpu :
lspci | grep -i nvidia

utilisation des gpu :
nvidia-smi


enter ipython :
grid5000

python-grid5000 : https://gitlab.inria.fr/msimonin/python-grid5000/-/tree/master?ref_type=heads