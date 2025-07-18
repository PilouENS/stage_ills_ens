#!/bin/bash

# Lancer le job OAR
cd $HOME/stage_ills_moe/stage_ILLS/pilou_git

JOB_ID=$(oarsub -l nodes=1/gpu=2,walltime=010:00:00 -q besteffort -p "gpu_model='A40'" "./run_on_node.sh" | grep -oP '(?<=OAR_JOB_ID=)\d+')

echo "Job soumis avec l'ID : $JOB_ID"
echo "En attente de la fin du job..."

while true; do
    STATUS=$(oarstat -s -j $JOB_ID)
    if [[ "$STATUS" == "$JOB_ID: Terminated" ]]; then
        echo "finito"
        break
    fi
    oarstat -u
    sleep 10
done

# Facultatif : afficher la sortie
# tail OAR.$JOB_ID.stdout
