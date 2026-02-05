#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=2:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb:scratch_local=8gb:cl_galdor=True
#PBS -N anlp02

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=/storage/plzen1/home/vladar21/nlp-cv03/run_cv03.py
ls /cvmfs/singularity.metacentrum.cz

#singularity run $CONTAINER pip install -r /storage/plzen1/nlp-cv01/requirements.txt --user

export WANDB_API_KEY="YOUR_WANDB_API_KEY"

singularity run --nv $CONTAINER python $PYTHON_SCRIPT --model $model --vocab_size $vocab_size --seq_len $seq_len --batches $batches --batch_size $batch_size --lr $lr --activation $activation --random_emb $random_emb --emb_training $emb_training --emb_projection $emb_projection --proj_size $proj_size --gradient_clip $gradient_clip --n_kernel $n_kernel --cnn_architecture $cnn_architecture

#PBS -l select=1:ncpus=1:ngpus=1:mem=40000mb:scratch_local=40000mb:cl_adan=True
#singularity run --nv $CONTAINER python $PYTHON_SCRIPT $ARG_1 $ARG_2
