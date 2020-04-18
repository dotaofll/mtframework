import logging
import os
import re
import unicodedata
from os.path import splitext
from collections import Counter

import numpy as np

import torch
from torch.utils.data import Dataset
from torchtext.utils import unicode_csv_reader

SOS_token = 0
EOS_token = 1


def StringWrapper(string: str):
    def opener(**kwargs):
        return string.format_map(kwargs)
    return opener


class BasicDataset(Dataset):

    if __debug__:
        base_data = StringWrapper(os.path.join(
            os.getcwd(), 'learn_pytorch', 'data', '{fuck}'))

    def __init__(self, filename: list):

        self.src_name = filename[0]
        self.tgt_name = filename[1]

        self.langs = [splitext(self.src_name)[1], splitext(self.tgt_name)[1]]

        def _read(filename):
            with open(filename, 'r', encoding='utf-8') as f_in:
                return [line.strip() for line in f_in]

        self.load_data = list(map(_read, [BasicDataset.base_data(
            fuck=self.src_name), BasicDataset.base_data(fuck=self.tgt_name)]))

        logging.info('Creating dataset with {name1} {name2}'.format(
            name1=self.src_name, name2=self.tgt_name))

    def __len__(self):
        return len(self.load_data[1])

    def map(self, func, num_of_workers: 0):
        pass

    def preprocess(self, idx, src_lang, tgt_lang, reverse=False):

        assert idx <= len(self.load_data[1])

        raw_pairs = [self.load_data[0][idx], self.load_data[1][idx]]

        pairs = [normalizeString(l) for l in raw_pairs]

        _src_lang, _tgt_lang = src_lang[1:], tgt_lang[1:]
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            #input_lang = Lang(_tgt_lang)
            #output_lang = Lang(_src_lang)
        # else:
            #input_lang = Lang(_src_lang)
            #output_lang = Lang(_tgt_lang)

        return pairs

    def build_vocab(self, filename=None, limit=0):
        src_vocab = {}
        tgt_vocab = {}
        limit = limit
        count_s = 0
        count_t = 0

        s_words, s_counts = self.__count_words('src')
        t_words, t_counts = self.__count_words('tgt')

        def control_symbols(string):
            if not string:
                return []
            else:
                return string.strip().split(',')

        ctrl_symbols = control_symbols("<pad>,<eos>,<unk>")
        for sym in ctrl_symbols:
            src_vocab[sym] = len(src_vocab)
            tgt_vocab[sym] = len(tgt_vocab)

        def _count(words, counts, vocab, count):
            for word, freq in zip(words, counts):
                if limit and len(vocab) >= limit:
                    break

                if word in vocab:
                    print("Warning: found duplicate token %s, ignored" % word)
                    continue

                vocab[word] = len(vocab)
                count += freq

        _count(s_words, s_counts, src_vocab, count_s)
        _count(t_words, t_counts, tgt_vocab, count_t)

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __count_words(self, direc: str):
        counter = Counter()
        if direc.lower() == 'src':
            data = self.load_data[0]
        else:
            data = self.load_data[1]
        for line in data:
            words = line.strip().split()
            counter.update(words)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, counts = list(zip(*count_pairs))
        return words, counts

    def __getitem__(self, idx):
        pairs = self.preprocess(
            idx, self.langs[0], self.langs[1], reverse=False)
        '''
        if input_lang.vocab is None and output_lang.vocab is None:
            input_lang.addSentence(pairs[0])
            output_lang.addSentence(pairs[1])
        '''
        '''
        self.input_tensor = tensorFromSentence(input_lang, pairs[0])
        self.target_tensor = tensorFromSentence(output_lang, pairs[1])
        '''
        return {'src': pairs[0],
                'tgt': pairs[1],
                'src_vocab': self.src_vocab,
                'tgt_vocab': self.tgt_vocab
                }
        # return {'input': self.input_tensor, 'target': self.target_tensor}


class Lang:
    def __init__(self, name, vocab: dict = None):
        self.name = name
        self.vocab = vocab
        #self.word2index = {}
        #self.word2count = {}
        self.index2word = {-1: "<unk>", 0: "<sos>", 1: "<eos>", 2: "<pad>"}
        self.n_words = 4
    '''
    def addSentence(self, sentence: str):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    '''

    def transword2index(self, word):
        if word in self.vocab:
            return self.vocab[word]
        return self.vocab['<unk>']


class Example(object):
    @classmethod
    def fromlist(cls, data, func):
        ex = cls()
        for val in data:
            val = func(val)
        return ex


def preprocess(data):
    pass


def unicode2Ascii(sentence):
    return ''.join(
        ch for ch in unicodedata.normalize('NFD', sentence)
        if unicodedata.category(ch) != 'Mn'
    )


def normalizeString(sentence):
    sentence = unicode2Ascii(sentence.lower().strip())
    sentence = re.sub(r'([.!?])', r' \1', sentence)
    sentence = re.sub(r'[^a-zA-Z.!?]+', r' ', sentence)
    return sentence


def indexesFromSentence(lang: Lang, sentence):
    if lang.vocab != None:
        return [lang.transword2index(word) for word in sentence.split(' ')]
    else:
        return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang: Lang, _dict, direc):
    get_pad_num = 'num_of_{}_pad'.format(direc)
    indexes = indexesFromSentence(lang, _dict[direc])
    pad_li = [lang.vocab['<pad>'] for x in range(_dict[get_pad_num]+1)]
    indexes.extend(pad_li)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device='cpu')
