#!/bin/bash

for i in {1..9}
do
    qsub -v random_emb="true",emb_training="false",emb_projection="true",final_metric="neural",vocab_size="30000",batch_size="1000",lr="0.1",optimizer="sgd",lr_scheduler="exponential" plan_cv02.sh
done

for i in {1..9}
do
    qsub -v random_emb="true",emb_training="false",emb_projection="true",final_metric="neural",vocab_size="20000",batch_size="500",lr="0.1",optimizer="sgd",lr_scheduler="step" plan_cv02.sh
done

for i in {1..9}
do
    qsub -v random_emb="true",emb_training="true",emb_projection="true",final_metric="neural",vocab_size="30000",batch_size="1000",lr="0.1",optimizer="sgd",lr_scheduler="exponential" plan_cv02.sh
done

for i in {1..9}
do
    qsub -v random_emb="true",emb_training="true",emb_projection="true",final_metric="neural",vocab_size="20000",batch_size="500",lr="0.1",optimizer="sgd",lr_scheduler="exponential" plan_cv02.sh
done