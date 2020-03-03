import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch import optim

from model.seq2seq.attention import Attention
from model.seq2seq.BaseRNN import BaseRNN


class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size, max_len, hidden_size, n_layers=1, rnn_cell='lstm',
                 input_dropout_p=.0, dropout_p=.0, embedding=None, update_embedding=True):
        super(EncoderRNN, self).__init__(vocab_size=vocab_size,
                                         max_len=max_len,
                                         hidden_size=hidden_size,
                                         n_layers=n_layers,
                                         rnn_cell=rnn_cell,
                                         input_dropout_p=input_dropout_p,
                                         dropout_p=dropout_p)

        self.rnn = rnn_cell(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            dropout=dropout_p)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=hidden_size)
        if embedding == None:
            self.embedding = nn.Embedding.from_pretrained(embedding)
        self.embedding.requires_grad = update_embedding

    def forward(self, input: torch.Tensor, input_length: None):
        embedded = self.embedding(input)
        output = self.dropout(embedded)
        output, hidden = self.rnn(output)
        return output, hidden


class DecoderRNN(BaseRNN):

    # static member
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size, n_layers=1, rnn_cell='lstm',
                 input_dropout_p=.0, dropout_p=.0, use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size=vocab_size,
                                         max_len=max_len,
                                         hidden_size=hidden_size,
                                         n_layers=n_layers,
                                         rnn_cell=rnn_cell,
                                         input_dropout_p=input_dropout_p,
                                         dropout_p=dropout_p)

        self.__output_size = vocab_size
        self.use_attention = use_attention

        self.embedding = nn.Embedding(
            num_embeddings=self.__output_size, embedding_dim=hidden_size)
        self.rnn = rnn_cell(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            dropout=dropout_p)

        if self.use_attention == True:
            self.attetion = Attention(dim=self.hidden_size)

        self.init_input = None

        self.out = nn.Linear(hidden_size, self.__output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward_step(self, input, hidden, encoder_outputs, activ_function):
        batch_size = input.size(0)
        output_size = input.size(1)
        output = self.embedding(input)
        output = self.dropout(output)
        output: torch.Tensor = activ_function(output)

        output, hidden = self.rnn(output, hidden)
        attn = None:
        if self.use_attention == True:
            output, attn = self.attetion(output=output,
                                         encoder_outputs=encoder_outputs)

        predict_softmax = self.softmax(self.out(output.contiguous(
        ).view(-1, self.hidden_size)).view(batch_size, output_size, -1))
        return predict_softmax, output, hidden

    def forward(self, input=None, encoder_hidden=None, encoder_outputs=None,
                function=F.relu, teacher_forcing_ratio=.5):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        if self.attetion == True:
            if encoder_outputs is None:
                raise ValueError()
        if input is None and encoder_hidden is None:
            batch_size = 1
        else:
            if input:
                batch_size = input.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)
        if input is None:
            if teacher_forcing_ratio > 0:
                raise ValueError(
                    "Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            input = torch.LongTensor(
                [0] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                input = input.cuda()
            max_length = self.max_len
        else:
            # minus the start of sequence symbol
            max_length = input.size(1) - 1

        decoder_hidden = encoder_hidden
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        if use_teacher_forcing:
            decoder_input = input[:, :, -1]
            decoder_output, decoder_hidden, attn = self.forward_step(input=decoder_input,
                                                                     hidden=decoder_hidden,
                                                                     encoder_outputs=encoder_outputs,
                                                                     activ_function=function)

            def decode(idx, step_out, step_attn):
                step_output = decoder_output[:, di, :]
                if attn:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decoder_outputs.append(step_output)
                if self.use_attention:
                    ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

                values, symbols = decoder_outputs[-1].topk(1)
                sequence_symbols.append(symbols)

                eos_batches = symbols.data.eq(1)
                if eos_batches.dim() > 0:
                    eos_batches = eos_batches.cpu().view(-1).numpy()
                    update_idx = ((lengths > step) & eos_batches) != 0
                    lengths[update_idx] = len(sequence_symbols)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)

        else:
            decoder_input = input[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden,
                                                                         encoder_outputs, function=function)

                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, input_length=None, target=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input=input,
                                                       input_length=input_length)

        result = self.decoder(input=target,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs)


def evaluate(model: nn.Module, sentence, max_length):
    with torch.no_grad():
        input_lang = utils.Lang('input')
        input_tensor = utils.tensorFromSentence(input_lang, sentence)

        input_length = input_tensor.size(0)
        encoder_hidden = torch.zeros(1, 1, model.hidden_size)

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for ei in range(input_length):
            output, hidden, attn_weights = model(
                input_tensor[ei], encoder_hidden)
