#!/bin/bash
#PBS -q default@pbs-m1.metacentrum.cz
#PBS -l walltime=100:00:0
#PBS -l select=1:ncpus=1:mem=16gb:scratch_local=128gb:cl_galdor=True
#PBS -N anlp04

CONTAINER=/cvmfs/singularity.metacentrum.cz/NGC/PyTorch:21.03-py3.SIF
PYTHON_SCRIPT=/storage/plzen1/home/vladar21/nlp-cv04/main04.py
ls /cvmfs/singularity.metacentrum.cz

##singularity run $CONTAINER pip install -r /storage/plzen1/home/vladar21/nlp-cv01/requirements.txt --user

singularity exec --writable-tmpfs $CONTAINER pip install torch==1.12.1+cpu -f https://download.pytorch.org/whl/torch_stable.html --user

singularity exec --writable-tmpfs $CONTAINER pip install "accelerate>=0.26.0" --user

singularity run $CONTAINER pip install transformers>=4.44.0 --user

singularity run $CONTAINER pip install seqeval --user

export WANDB_API_KEY="YOUR_WANDB_API_KEY"

#export PYTHONUSERBASE=/storage/plzen1/home/vladar21/my_pip_libs
#export PATH=$PYTHONUSERBASE/bin:$PATH

if [[ "$model_type" == "CZERT" ]]; then
singularity run $CONTAINER python3 $PYTHON_SCRIPT --model_type "CZERT" --freeze_first_x_layers $freeze_layer --learning_rate 0.0001 --l2_alpha 0.01 --data_dir "/storage/plzen1/home/vladar21/nlp-cv04/data" --labels "/storage/plzen1/home/vladar21/nlp-cv04/data/labels.txt" --eval_steps 100 --num_train_epochs 50 --task "NER" --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --output_dir output --do_predict --do_train --do_eval --eval_dataset_batches 200 --logging_steps 50 --warmup_steps 4000 --dropout_probs 0.05 --lstm_hidden_dimension 128 --num_lstm_layers 2 --embedding_dimension 128

singularity run $CONTAINER python3 $PYTHON_SCRIPT --freeze_embedding_layer --model_type "CZERT" --freeze_first_x_layers $freeze_layer --learning_rate 0.0001 --l2_alpha 0.01 --data_dir "/storage/plzen1/home/vladar21/nlp-cv04/data" --labels "/storage/plzen1/home/vladar21/nlp-cv04/data/labels.txt" --eval_steps 100 --num_train_epochs 50 --task "NER" --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --output_dir output --do_predict --do_train --do_eval --eval_dataset_batches 200 --logging_steps 50 --warmup_steps 4000 --dropout_probs 0.05 --lstm_hidden_dimension 128 --num_lstm_layers 2 --embedding_dimension 128
fi

if [[ "$model_type" == "BERT" ]]; then
    singularity run $CONTAINER python3 $PYTHON_SCRIPT --model_type "BERT" --learning_rate $lr --l2_alpha $l2_alpha --data_dir "/storage/plzen1/home/vladar21/nlp-cv04/data" --labels "/storage/plzen1/home/vladar21/nlp-cv04/data/labels.txt" --eval_steps 100 --num_train_epochs 50 --task "NER" --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --output_dir output --do_predict --do_train --do_eval --eval_dataset_batches 200 --logging_steps 50 --warmup_steps 4000 --dropout_probs 0.05 --lstm_hidden_dimension 128 --num_lstm_layers 2 --embedding_dimension 128

    singularity run $CONTAINER python3 $PYTHON_SCRIPT --freeze_embedding_layer --model_type "BERT" --learning_rate $lr --l2_alpha $l2_alpha --data_dir "/storage/plzen1/home/vladar21/nlp-cv04/data" --labels "/storage/plzen1/home/vladar21/nlp-cv04/data/labels.txt" --eval_steps 100 --num_train_epochs 50 --task "NER" --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --output_dir output --do_predict --do_train --do_eval --eval_dataset_batches 200 --logging_steps 50 --warmup_steps 4000 --dropout_probs 0.05 --lstm_hidden_dimension 128 --num_lstm_layers 2 --embedding_dimension 128
fi

#PBS -l select=1:ncpus=1:ngpus=1:mem=40000mb:scratch_local=40000mb:cl_adan=True
#singularity run --nv $CONTAINER python $PYTHON_SCRIPT $ARG_1 $ARG_2
