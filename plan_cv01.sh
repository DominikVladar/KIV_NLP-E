#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=1:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=4000mb:scratch_local=1000mb:cl_adan=True
#PBS -N anlp01

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=/storage/plzen1/home/vladar21/nlp-cv01/run_cv01.py
ls /cvmfs/singularity.metacentrum.cz

#singularity run $CONTAINER pip install -r /storage/plzen1/nlp-cv01/requirements.txt --user

export WANDB_API_KEY="YOUR_WANDB_API_KEY"

singularity run --nv $CONTAINER python $PYTHON_SCRIPT --model $model --optimizer $optimizer --lr $lr --dp $dp

#PBS -l select=1:ncpus=1:ngpus=1:mem=40000mb:scratch_local=40000mb:cl_adan=True
#singularity run --nv $CONTAINER python $PYTHON_SCRIPT $ARG_1 $ARG_2
