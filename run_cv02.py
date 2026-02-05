from cv02.main02 import main
import argparse


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() == 'false':
        return False
    elif value.lower() == 'true':
        return True
    raise ValueError(f'{value} is not a valid boolean value')

if __name__ == '__main__':
    """    main({
        'vocab_size': 20000,
        'batch_size': 1000,
        'lr': 0.01,
        'optimizer': 'adam',
        'lr_scheduler': 'step',
        'random_emb': False,
        'emb_training': True,
        'emb_projection': True,
        'final_metric': 'cos',
    })"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_scheduler', type=str, default='step')
    parser.add_argument('--random_emb', type=str, default='false')
    parser.add_argument('--emb_training', type=str, default='true')
    parser.add_argument('--emb_projection', type=str, default='true')
    parser.add_argument('--final_metric', type=str, default='cos')

    args = parser.parse_args()
    main({
        'vocab_size': args.vocab_size,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'optimizer': args.optimizer,
        'lr_scheduler': args.lr_scheduler,
        'random_emb': str_to_bool(args.random_emb),
        'emb_training': str_to_bool(args.emb_training),
        'emb_projection': str_to_bool(args.emb_projection),
        'final_metric': args.final_metric,
    })
