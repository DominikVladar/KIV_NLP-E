#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=20:00:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb:scratch_local=128gb:cl_galdor=True
#PBS -N anlp04

export PATH=$HOME/.local/bin:$PATH

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:25.02-py3.SIF
PYTHON_SCRIPT=/storage/plzen1/home/vladar21/nlp-cv04/main04.py
ls /cvmfs/singularity.metacentrum.cz

##singularity run $CONTAINER pip install -r /storage/plzen1/home/vladar21/nlp-cv01/requirements.txt --user

singularity exec --nv $CONTAINER pip install wandb --user

singularity exec --nv $CONTAINER pip install seqeval --user

singularity exec --nv $CONTAINER pip install transformers[torch] --user

export WANDB_API_KEY="YOUR_WANDB_API_KEY"

#export PYTHONUSERBASE=/storage/plzen1/home/vladar21/my_pip_libs
#export PATH=$PYTHONUSERBASE/bin:$PATH


singularity exec --nv $CONTAINER python3 $PYTHON_SCRIPT $args


#PBS -l select=1:ncpus=1:ngpus=1:mem=40000mb:scratch_local=40000mb:cl_adan=True
#singularity run --nv $CONTAINER python $PYTHON_SCRIPT $ARG_1 $ARG_2
