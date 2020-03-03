import os
from collections import defaultdict

import torch
import torchtext as tt
from torch.utils.data import DataLoader, random_split

import learn_pytorch.utils.DataSet as ud
import tensorflow as tf


SRC_VOCAB = None
TGT_VOCAB = None

INPUT_LANG = None
OUTPUT_LANG = None

def pad(batch: list):
    # batch is like [pairs]
    # pairs are like [src_seq,tgt_seq]
    smxlen = 0
    tmxlen = 0

    for pair in batch:
        if smxlen < len(pair[0].split(' ')):
            smxlen = len(pair[0].split(' '))
        if tmxlen < len(pair[1].split(' ')):
            tmxlen = len(pair[1].split(' '))

    src_padded = []
    tgt_padded = []

    for x in batch:
        src = x[0]
        tgt = x[1]

        '''
        padded.append(
            ['<pad>'] * max(0, smxlen - len(x))
            + ([])
            + list(x[:smxlen])
            + ([]))
        '''
        batch_dict = {
            'src': src,
            'tgt': tgt,
            'num_of_src_pad': max(0, smxlen-len(src.split(' '))),
            'num_of_tgt_pad': max(0, tmxlen-len(tgt.split(' ')))
        }

        src_padded.append(ud.tensorFromSentence(INPUT_LANG, batch_dict, 'src'))
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

    name=[src,tgt]

    data = ud.BasicDataset(name)

    SRC_VOCAB, TGT_VOCAB = data.build_vocab(limit=16000)

    src_dataset = tf.data.TextLineDataset(src_dir)
    tgt_dataset = tf.data.TextLineDataset(tgt_dir)

    iterator = src_dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))

    dataset = tf.data.Dataset.zip((src_dataset,tgt_dataset))

    dataset = dataset.map(
            lambda src, tgt: (
                tf.string_split([src]).values,
                tf.string_split([tgt]).values
            ),
            num_parallel_calls=4
        )

    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        for i in range(5):
            print(sess.run(one_element))

        # Append <eos> symbol
    dataset = dataset.map(
        lambda src, tgt: (
            tf.concat([src, [tf.constant(0)]], axis=0),
            tf.concat([tgt, [tf.constant(0)]], axis=0)
        ),
        num_parallel_calls=4
    )

        # Convert to dictionary
    dataset = dataset.map(
        lambda src, tgt: {
            "source": src,
            "target": tgt,
            "source_length": tf.shape(src),
            "target_length": tf.shape(tgt)
        },
        num_parallel_calls=4
    )

        # Create iterator
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

        # Create lookup table
    src_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(SRC_VOCAB),
        default_value=SRC_VOCAB['<unk>']
    )
    tgt_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(TGT_VOCAB),
        default_value=TGT_VOCAB['<unk>']
    )

        # String to index lookup
    features["source"] = src_table.lookup(features["source"])
    features["target"] = tgt_table.lookup(features["target"])


    print(1)
    '''
    data = ud.BasicDataset(name)

    SRC_VOCAB, TGT_VOCAB = data.build_vocab(limit=16000)
    
    INPUT_LANG = ud.Lang('src',SRC_VOCAB)
    OUTPUT_LANG = ud.Lang('tgt',TGT_VOCAB)

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
        print("get train source input: {input_size} target input: {target_size} ".format(input_size=input.size(),target_size=target.size()))
    print(1)
    '''