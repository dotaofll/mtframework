import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import DataSet
import model.seq2seq.seq2seq_model as models
from DataSet import EOS_token, SOS_token, _device

TEACHER_FORCING_RATIO = .5

src_dir = 'data/src/'
tgt_dir = 'data/tgt/'


def train_seq(input_tensor: torch.Tensor, target_tensor: torch.Tensor, encoder: nn.Module, decoder: nn.Module, encoder_optimizer, decoder_optimizer,
              criterion, max_length=10, lr=0.1):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        max_length, encoder.hidden_size, device=_device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden
        )
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=_device)
    decoder_hidden = encoder_hidden

    using_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

    if using_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

        loss += criterion(decoder_output, target_tensor[di])
    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length


def trainIters(encoder:nn.Module, decoder:nn.Module, n_iters, print_every=1000, val_persent=.1, plot_every=100, learning_rate=0.01):
    start = utils.timer.time()

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

            loss = train_seq(input_tensor, target_tensor, encoder,
                             decoder, encoder_optimizer, decoder_optimizer, criterion)

            print_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
