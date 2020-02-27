import torch.nn as nn


class BaseRNN(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, hidden_size: int,
                 n_layers: int, rnn_cell: str, input_dropout_p: float, dropout_p: float):
        super(BaseRNN, self).__init__()

        self.__vocab_size = vocab_size
        self.__max_len = max_len
        self.__hidden_size = hidden_size
        self.__n_layers = n_layers
        self.__input_dropout_p = input_dropout_p
        
        self.dropout = nn.Dropout(dropout_p)
        
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        else if rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
