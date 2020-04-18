import os
from collections import defaultdict

import torch
import torchtext as tt
from torch.utils.data import DataLoader, random_split

import learn_pytorch.utils.DataSet as ud
import tensorflow as tf


def pad(batch: dict):
    # batch is like [pairs]
    # pairs are like [src_seq,tgt_seq]

    smxlen = 0
    tmxlen = 0

    class tok():
        def __init__(self,str,tokenizer:None):
            if tokenizer is None:
                self.token = str.split(' ')
        def __len__(self):
            return len(self.token)

    for src_seq, tgt_seq, _s, _t in batch.values():
        maxsrc_seq = max(smxlen,len(tok(src_seq)))
        maxtgt_seq = max(tmxlen,len(tok(tgt_seq)))

    src_padded = []
    tgt_padded = []

    for x in batch:
        src = x[0]
        tgt = x[1]

        batch_dict = {
            'src': src,
            'tgt': tgt,
            'num_of_src_pad': max(0, smxlen-len(src.split(' '))),
            'num_of_tgt_pad': max(0, tmxlen-len(tgt.split(' ')))
        }

        src_padded.append(ud.tensorFromSentence(
            INPUT_LANG, batch_dict, 'src'))
        tgt_padded.append(ud.tensorFromSentence(
            OUTPUT_LANG, batch_dict, 'tgt'))

    return {'input': torch.stack(src_padded), 'target': torch.stack(tgt_padded)}


if __name__ == "__main__":
    cwd = os.getcwd()
    file = os.path.join(cwd, 'learn_pytorch', 'data', 'tr2zh')
    src_dir = os.path.join(cwd, 'learn_pytorch', 'data', 'tr2zh.tr')
    tgt_dir = os.path.join(cwd, 'learn_pytorch', 'data', 'tr2zh.zh')

    src = 'tr2zh.tr'
    tgt = 'tr2zh.zh'

    with open('learn_pytorch\\data\\tr2zh.tr') as fin:
        pass

    name = [src, tgt]

    test_s = ["i want to meet you"]
    test_t = ["jdiaboo dajoi da"]

    data = ud.BasicDataset(name)
    data.build_vocab(limit=16000)

    n_val = int(len(data)*.2)
    n_train = len(data) - n_val
    trainPairs, validPairs = random_split(data, [n_train, n_val])
    train_loader = DataLoader(data, batch_size=1024, collate_fn=pad,
                              shuffle=True, num_workers=4)
    # val_loader = DataLoader(validPairs, batch_size=64,
    #                        shuffle=True, num_workers=8, pin_memory=True)
    for batch in train_loader:
        input = batch['input']
        target = batch['target']
        print("get train source input: {input_size} target input: {target_size} ".format(
            input_size=input.size(), target_size=target.size()))
    print(1)
