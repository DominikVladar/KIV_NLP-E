#!/bin/bash


qsub -v A=3,B=9 /storage/brno1-cerit/home/sidoj/anlp/cv-01/run_hello_on_meta.sh
qsub -v A=2,B=5 /storage/brno1-cerit/home/sidoj/anlp/cv-01/run_hello_on_meta.sh
qsub -v A=1,B=7 /storage/brno1-cerit/home/sidoj/anlp/cv-01/run_hello_on_meta.sh


