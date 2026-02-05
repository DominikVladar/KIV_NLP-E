#!/bin/bash
#PBS -q default@meta-pbs.metacentrum.cz
#PBS -l walltime=1:0:0
#PBS -l select=1:ncpus=1:mem=400mb:scratch_local=400mb
#PBS -N anlp01-hello

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=/storage/brno1-cerit/home/sidoj/anlp/cv-01/hello_world_meta.py

ls /cvmfs/singularity.metacentrum.cz

#singularity run $CONTAINER pip install wandb --user
#singularity run $CONTAINER python -m wandb login --relogin c1246....3c992bf5

singularity run $CONTAINER python $PYTHON_SCRIPT -a 4 -b 5
#singularity run $CONTAINER python $PYTHON_SCRIPT -a $A -b $B

#PBS -l select=1:ncpus=1:ngpus=1:mem=40000mb:scratch_local=40000mb:cl_adan=True
#singularity run --nv $CONTAINER python $PYTHON_SCRIPT $ARG_1 $ARG_2
