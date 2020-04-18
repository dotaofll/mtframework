from __future__ import print_function, division

import torch
import torchtext

import model.seq2seq
from loss.loss import NLLLoss


class Evaluator(object):
    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
        '''
        Args:
            model(seq2seq.models)
            data(dataset.fields)
        Returns:
            loss(float)
        '''
        model.eval()
        loss = self.loss
        loss.reset()

        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iter = torchtext.data.BucketIterator(dataset=data,
                                                   batch_size=self.batch_size,
                                                   sort_key=lambda x: len(x.src),
                                                   device=device,
                                                   train=False)

        tgt_vocab = data.fields['tgt'].vocab
        pad_seg = tgt_vocab.stoi[data.fields['tgt'].pad_token]

        with torch.no_grad():
            for batch in batch_iter:
                input_var, input_length = batch.src
                target_var = batch.tgt

                decoder_outputs, decoder_hidden, other = model(
                    input_var, input_length, target_var)

                seqlist = other['sequence']
                for step, step_output in enumerate(decoder_outputs):
                    target = target_var[:, step+1]
                    loss.eval_batch(step_output.view(
                        target_var.size(0), -1), target)

                    non_padding = target.ne(pad_seg)
                    correct = seqlist[step].view(-1).eq(
                        target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()
                if total == 0:
                    accuracy = float('nan')
                else:
                    accuracy = match/total

                return loss.get_loss(), accuracy
