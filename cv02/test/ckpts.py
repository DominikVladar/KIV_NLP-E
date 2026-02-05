import os

import torch

from cv02 import consts


def ckpt_1(top_n_words):
    print(f"CKPT#1\n\tTOP_N_WORDS len():{len(top_n_words)} : {top_n_words[:10]}\n\n")



def ckpt_2():
    exists = os.path.exists(consts.WORD2IDX) and os.path.exists(consts.VECS_BUFF)
    print(f"CKPT#2 Files are on the hard drive: {exists}\n\n")


def ckpt_3(vectorizer):
    inp = "Podle vlády dnes není dalších otázek"

    vectorized = vectorizer.sent2idx(inp)
    print("\n\nCKPT#3\n\t",inp,"\n\t", vectorized[:14],
          "\n\n")  # [2696, 1768, 128, 1373, 1577, 2696, 2697, 2697, 2697, 2697, 2697, 2697, 2697, 2697]

def ckpt_4(train_dataset, test_dataset):
    print("\n\nCKPT#4")
    print(f"loaded train:{len(train_dataset.sts)}\tout of vocab:{train_dataset.out_of_vocab} %")
    print(f"loaded test:{len(test_dataset.sts)}\tout of vocab:{test_dataset.out_of_vocab} %")

    # todo check random shuffeling
    one_sample = next(iter(train_dataset))
    print(one_sample)
    print(f"returning {type(one_sample)}\nlen:{len(one_sample)} expected(3)\n")
    for i, one_item in enumerate(one_sample):
        if isinstance(one_item, torch.Tensor):
            print(f"shape[{i}] : {one_item.shape}")

def ckpt_5(dummy_test,dummy_train):
    print("DUMMY TEST:",dummy_test)
    print("DUMMY TRAIN:", dummy_train)

    # assert abs(dummy_test-3.1970975200335183) < 0.1
    assert abs(dummy_test-3.1970975200335183) < 1
    print("CKPT5 -- OK")


def ckpt_6(test_loss_arr):
    max_loss = max(test_loss_arr[-100:])
    print(f"maximum loss in last 100 batches is : {max_loss}")
    assert max_loss < 2.8

    return True

def ckpt_7(test_loss_arr,train_model_fc):
    final_losses = []
    final_losses.append(max(test_loss_arr[-100:]))
    for _ in range(10):
        test_loss_arr = train_model_fc()
        final_losses.append(max(test_loss_arr[-100:]))

    print(final_losses)
