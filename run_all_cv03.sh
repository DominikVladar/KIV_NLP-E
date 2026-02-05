#!/bin/bash
for batch_size in 64 128
do
    for lr in 0.001 0.0001 0.00001 0.000001
    do
        for activation in relu gelu
        do
            for model in cnn mean
            do
                for random_emb in true false
                do
                    for emb_training in true false
                    do
                        for emb_projection in true false
                        do
                            for cnn_architecture in A B C
                            do
                                        qsub -v model="$model",vocab_size="20000",seq_len="100",batches="5000",batch_size="$batch_size",lr="$lr",activation="$activation",random_emb="$random_emb",emb_training="$emb_training",emb_projection="$emb_projection",proj_size="100",gradient_clip="0.5",n_kernel="64",cnn_architecture="$cnn_architecture" plan_cv03.sh
                            done
                        done
                    done
                done
            done
        done
    done
done