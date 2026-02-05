for iteration in 1 2 3 4 5 6 7 8 9 10
do

    qsub -v model="cnn",vocab_size="20000",seq_len="100",batches="100000",batch_size="32",lr="0.0001",activation="gelu",random_emb="true",emb_training="true",emb_projection="true",proj_size="128",gradient_clip="0.5",n_kernel="64",cnn_architecture="A" plan_cv03.sh

    qsub -v model="cnn",vocab_size="20000",seq_len="100",batches="5000",batch_size="64",lr="0.0001",activation="relu",random_emb="false",emb_training="true",emb_projection="true",proj_size="100",gradient_clip="0.5",n_kernel="64",cnn_architecture="B" plan_cv03.sh

    qsub -v model="cnn",vocab_size="20000",seq_len="100",batches="5000",batch_size="128",lr="0.0001",activation="relu",random_emb="false",emb_training="true",emb_projection="true",proj_size="100",gradient_clip="0.5",n_kernel="64",cnn_architecture="C" plan_cv03.sh

    qsub -v model="mean",vocab_size="20000",seq_len="100",batches="5000",batch_size="64",lr="0.001",activation="relu",random_emb="true",emb_training="true",emb_projection="true",proj_size="100",gradient_clip="0.5",n_kernel="64",cnn_architecture="A" plan_cv03.sh

done