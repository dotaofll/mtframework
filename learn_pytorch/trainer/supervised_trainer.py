from __future__ import division

import logging
import os
import random
import time

import torch
import torchtext
from torch import optim

import model.seq2seq as seq2seq
from evaluator.evaluator import Evaluator
from loss.loss import NLLLoss
from optim.optimer import Optimizer
from util.checkpoint import Checkpoint


class SupervisedTrainer(object):
    def __init__(self, export_dir='experiment',
                 loss=NLLLoss(),
                 batch_size=64,
                 random_seed=None,
                 checkpoint_every=100,
                 print_every=100):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(export_dir):
            export_dir = os.path.join(os.getcwd(), export_dir)
        self.export_dir = export_dir
        if not os.path.exists(self.export_dir):
            os.makedirs(self.export_dir)
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio=.5):
        loss = self.loss
        loss.reset()

        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable)

        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)
            loss.eval_batch(step_output.contiguous().view(
                batch_size, -1), target_variable[:, step+1])

        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data, teacher_forcing_ratio=0):
        log = self.logger
        print_loss_total = 0
        epoch_loss_total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(dataset=data, batch_size=self.batch_size,
                                                       sort=False, sort_within_batch=True,
                                                       sort_key=lambda x: len(x.src),
                                                       device=device, repeat=False)

        step_per_epoch = len(batch_iterator)

        total_steps = step_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs+1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()
            for _ in range((epoch - 1) * step_per_epoch, step):
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1
                input_vat,input_length = getattr(batch,'src')
                target_var = batch.tgt

                loss = self._train_batch(
                    input_variable=input_vat,
                    input_lengths=input_length,
                    target_variable=target_var,
                    model=model)

                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    log.info(log_msg)
                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               input_vocab=data.fields['src'].vocab,
                               output_vocab=data.fields['tgt'].vocab).save(self.export_dir)

            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / \
                min(step_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f" % (
                epoch, self.loss.name, epoch_loss_avg)
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (
                    self.loss.name, dev_loss, accuracy)
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=0):

        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(
                self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('param', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(
                model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step

        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(
                    model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" %
                         (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data, model, num_epochs,
                            start_epoch, step, dev_data=dev_data,
                            teacher_forcing_ratio=teacher_forcing_ratio)
        return model
