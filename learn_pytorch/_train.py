import argparse
import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torchtext as tt
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import learn_pytorch.utils.DataSet
import learn_pytorch.model.seq2seq.seq2seq_model as models

import learn_pytorch.utils.timer
from DataSet import EOS_token, SOS_token, _device
from model.seq2seq.seq2seq_model import Seq2Seq

TEACHER_FORCING_RATIO = .5



def train_seq(input_tensor: torch.Tensor, target_tensor, models: Seq2Seq, decoder: nn.Module, encoder_optimizer, decoder_optimizer,
              criterion, max_length=10, lr=0.1):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=_device)

    loss = 0

    for ei in range(input_length):
        decoder_outputs, decoder_hidden, other = models(input=input_tensor,
                                                        input_length=input_length,
                                                        target=target_tensor,
                                                        teacher_forcing_ratio=TEACHER_FORCING_RATIO)

    for step, step_output in enumerate(decoder_outputs):
        batch_size = target_length
        loss += criterion(step_output.contiguous().view(batch_size, -1),
                          target_tensor[:, step + 1])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length


def trainIters(model: Seq2Seq, n_iters, print_every=1000, val_persent=.1, plot_every=100, learning_rate=0.01):
    start = learn_pytorch.utils.timer.time()

    print_loss_total = 0
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    dataset = DataSet.BasicDataset(src_dir, tgt_dir)

    n_val = int(len(dataset) * val_persent)
    n_train = len(dataset) - n_val

    trainPairs, validPairs = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(trainPairs, batch_size=64,
                              shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(validPairs, batch_size=64,
                            shuffle=True, num_workers=8, pin_memory=True)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters+1):
        for batch in train_loader:
            input_tensor = batch['input']
            target_tensor = batch['target']

            loss = train_seq(input_tensor, target_tensor, model,
                             encoder_optimizer, decoder_optimizer, criterion)

            print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (learn_pytorch.utils.timer.timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
