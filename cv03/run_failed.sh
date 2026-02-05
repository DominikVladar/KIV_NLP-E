for iteration in 1 2
do
    for random_emb in true false
    do
        for emb_training in true false
        do
            qsub -v model="cnn",vocab_size="20000",seq_len="100",batches="100000",batch_size="32",lr="0.0001",activation="gelu",random_emb="$random_emb",emb_training="$emb_training",emb_projection="true",proj_size="128",gradient_clip="0.5",n_kernel="64",cnn_architecture="A" plan_cv03.sh
        done
    done
done