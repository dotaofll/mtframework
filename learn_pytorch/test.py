import argparse
import logging
import os

import torch
import torchtext
from torch.optim.lr_scheduler import StepLR

import model.seq2seq as seq2seq
from dataset.fields import SourceField, TargetField
from evaluator.preditor import Predictor
from loss.loss import Perplexity
from model.seq2seq.seq2seq_model import DecoderRNN, EncoderRNN,Seq2Seq
from optim.optimer import Optimizer
from trainer.supervised_trainer import SupervisedTrainer
from util.checkpoint import Checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',default='data/corpus',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',default='data/dev',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
args = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(
    logging, args.log_level.upper()))
logging.info(args)

if args.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(
        args.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, args.load_checkpoint)))
    checkpoint_path = os.path.join(
        args.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, args.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab

else:
    src = SourceField()
    tgt = TargetField()
    max_len = 50

    def len_filter(sample):
        return len(sample.src) <= max_len and len(sample.tgt) <= max_len

    train = torchtext.datasets.TranslationDataset(path=args.train_path,exts=('.en','.zh'),fields=[('src',src),('tgt',tgt)],filter_pred=len_filter)
    dev = torchtext.datasets.TranslationDataset(path=args.dev_path,exts=('.en','.zh'),fields=[('src',src),('tgt',tgt)],filter_pred=len_filter)

    src.build_vocab(train,max_size=500)
    tgt.build_vocab(train,max_size=500)

    input_vocab=src.vocab
    output_vocab = tgt.vocab

    #prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight,pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq_model = None
    optimizer = None
    if not args.resume:
        hidden_size = 128

        encoder = EncoderRNN(len(input_vocab),max_len,hidden_size,rnn_cell='gru')
        decoder = DecoderRNN(len(output_vocab),max_len,hidden_size,use_attention=True,rnn_cell='gru',eos_id=tgt.eos_id, sos_id=tgt.sos_id)

        seq2seq_model = Seq2Seq(encoder,decoder)

        if torch.cuda.is_available():
            seq2seq_model.cuda()

        for param in seq2seq_model.parameters():
            param.data.uniform_(-0.08,0.08)

    t = SupervisedTrainer(loss=loss, batch_size=32,
                          checkpoint_every=50,
                          print_every=10, export_dir=args.expt_dir)

    seq2seq = t.train(seq2seq_model, train,
                      num_epochs=6, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=args.resume)

predictor = Predictor(seq2seq_model, input_vocab, output_vocab)

while True:
    seq_str = input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
