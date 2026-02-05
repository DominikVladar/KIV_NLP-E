import argparse
import torch

from cv03.main03 import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", type=str, default="cnn")
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--batches", type=int, default=500000)
    parser.add_argument("--batch_size", type=int, default=33)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--random_emb", type=str, default="false")
    parser.add_argument("--emb_training", type=str, default="true")
    parser.add_argument("--emb_projection", type=str, default="true")
    parser.add_argument("--proj_size", type=int, default=128)
    parser.add_argument("--gradient_clip", type=float, default=.5)
    parser.add_argument("--n_kernel", type=int, default=64)
    parser.add_argument("--cnn_architecture", type=str, default="A")

    args = parser.parse_args()

    config = {
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "model": args.model,

        "vocab_size": args.vocab_size,
        "seq_len": args.seq_len,

        "batches": args.batches,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "activation": args.activation,
        "random_emb": args.random_emb == "true",
        "emb_training": args.emb_training == "true",
        "emb_projection": args.emb_projection == "true",
        "proj_size": args.proj_size,
        "gradient_clip": args.gradient_clip,

        "n_kernel": args.n_kernel,
        "cnn_architecture": args.cnn_architecture
    }

    main(config)