from cv01.main01 import main
import argparse

if __name__ == '__main__':
    """config = {
        "lr": 0.01,
        "use_normalization":False,
        "optimizer": "sgd", # ADAM,
        "batch_size":10,
        "dp":0,
        "scheduler":"exponential",
        "gamma":0.9,
        #"step_size":5
    }
"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--model', type=str, default='dense')
    parser.add_argument('--dp', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='sgd')

    args = parser.parse_args()

    main({
        "lr": args.lr,
        "use_normalization":False,
        "optimizer": args.optimizer,
        "batch_size":10,
        "dp": args.dp,
        "scheduler":"exponential",
        "gamma":0.9,
        "model": args.model,
        #"step_size":5
    })
    # main()
