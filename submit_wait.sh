#!/bin/bash

# Lancer le job OAR
JOB_ID=$(oarsub -l gpu=1,walltime=00:20:00 -q besteffort -p "gpu_model='H100 NVL'" "./run_on_node.sh" | grep -oP '(?<=OAR_JOB_ID=)\d+')

echo "Job soumis avec l'ID : $JOB_ID"
echo "En attente de la fin du job..."


# Facultatif : afficher la sortie
# tail OAR.$JOB_ID.stdout
