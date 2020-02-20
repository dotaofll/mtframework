import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=_device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=_device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1
        )
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=_device)


class Encoder_Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(Encoder_Seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = num_layers
        self.encoders = nn.ModuleList(
            [EncoderRNN(self.input_size, self.hidden_size) for layer in range(self.layers)])

    def forward(self, input, hidden):
        for encoder in self.encoders:
            output, hidden = encoder(input, hidden)
        return output, hidden


class Decoder_Seq(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=4, dropout=.1, max_len=10, use_atten=True):
        super(Decoder_Seq, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = num_layers
        self.dropout = dropout
        self.maxLen = max_len

        if use_atten == True:
            self.decoders = nn.ModuleList([AttnDecoderRNN(
                self.hidden_size, self.output_size, self.dropout, self.maxLen) for layer in range(self.layers)])

        self.decoders = nn.ModuleList(
            [DecoderRNN(self.hidden_size, self.output_size) for layer in range(self.layers)])

    def forward(self, input, hidden, encoder_outputs):
        for decoder in self.decoders:
            output, hidden, attn_weights = decoder(
                input, hidden, encoder_outputs)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, dropout=.1, max_len=10, use_atten=True):
        super(Seq2Seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = num_layers
        self.dropout = dropout
        self.maxLen = max_len

        self.encoder = Encoder_Seq(
            self.input_size, self.hidden_size, self.layers)
        self.decoder = Decoder_Seq(
            self.hidden_size, self.output_size, self.layers, self.dropout, self, max_len, use_atten)

    def forward(self, input, hidden):
        output, hidden = self.encoder(input, hidden)
        output, hidden, attn_weights = self.decoder(input, hidden, output)
        return output, hidden, attn_weights
