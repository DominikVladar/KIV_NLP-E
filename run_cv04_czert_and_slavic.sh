#!/bin/bash


qsub -v args="--per_device_train_batch_size 32 --per_device_eval_batch_size 32 --model_type CZERT --data_dir /storage/plzen1/home/vladar21/nlp-cv04/data --labels /storage/plzen1/home/vladar21/nlp-cv04/data/labels.txt --output_dir /storage/plzen1/home/vladar21/nlp-cv04/output --do_predict --do_train --do_eval --eval_steps 100 --logging_steps 50 --learning_rate 0.0001 --warmup_steps 4000 --num_train_epochs 50 --dropout_probs 0.05 --l2_alpha 0.01 --task NER" plan_base_cv04.sh

qsub -v args="--per_device_train_batch_size 32 --per_device_eval_batch_size 32 --model_type SLAVIC --data_dir /storage/plzen1/home/vladar21/nlp-cv04/data --labels /storage/plzen1/home/vladar21/nlp-cv04/data/labels.txt --output_dir /storage/plzen1/home/vladar21/nlp-cv04/output --do_predict --do_train --do_eval --eval_steps 100 --logging_steps 50 --learning_rate 0.0001 --warmup_steps 4000 --num_train_epochs 50 --dropout_probs 0.05 --l2_alpha 0.01 --task NER" plan_base_cv04.sh

qsub -v args="--per_device_train_batch_size 32 --per_device_eval_batch_size 32 --model_type CZERT --data_dir /storage/plzen1/home/vladar21/nlp-cv04/data-mt --labels /storage/plzen1/home/vladar21/nlp-cv04/data-mt/labels.txt --output_dir /storage/plzen1/home/vladar21/nlp-cv04/output --do_predict --do_train --do_eval --eval_steps 300 --logging_steps 50 --learning_rate 0.0001 --warmup_steps 4000 --num_train_epochs 10 --dropout_probs 0.05 --l2_alpha 0.01 --task TAGGING" plan_base_cv04.sh

qsub -v args="--per_device_train_batch_size 32 --per_device_eval_batch_size 32 --model_type SLAVIC --data_dir /storage/plzen1/home/vladar21/nlp-cv04/data-mt --labels /storage/plzen1/home/vladar21/nlp-cv04/data-mt/labels.txt --output_dir /storage/plzen1/home/vladar21/nlp-cv04/output --do_predict --do_train --do_eval --eval_steps 300 --logging_steps 50 --learning_rate 0.0001 --warmup_steps 4000 --num_train_epochs 10 --dropout_probs 0.05 --l2_alpha 0.01 --task TAGGING" plan_base_cv04.sh
