#!/bin/bash

for random_emb in true false
do
    for emb_training in true false
    do
        for emb_projection in true false
        do
            for final_metric in cos neural
            do
                for vocab_size in 20000 30000
                do
                    for batch_size in 500 1000
                    do
                        for lr in 0.1 0.01 0.001 0.0001 0.00001
                        do
                            for optimizer in sgd adam
                            do
                                for lr_scheduler in step multistep exponential
                                do
                                    qsub -v random_emb="$random_emb",emb_training="$emb_training",emb_projection="$emb_projection",final_metric="$final_metric",vocab_size="$vocab_size",batch_size="$batch_size",lr="$lr",optimizer="$optimizer",lr_scheduler="$lr_scheduler" plan_cv02.sh
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done