import logging
import os
import re
import unicodedata
from os.path import splitext

import numpy as np

import torch
from torch.utils.data import Dataset

SOS_token = 0
EOS_token = 1


class BasicDataset(Dataset):
    def __init__(self, src_dir: str, tgt_dir: str):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir

        def spilt_name(_str): return _str[_str.rfind(os.sep)+1:]
        self.filename = [spilt_name(src_dir), spilt_name(tgt_dir)]
        self.langs = [splitext(src_dir)[1], splitext(tgt_dir)[1]]

        logging.info('Creating dataset with {name1} {name2}'.format(name1=self.filename[0], name2=self.filename[1]))

    def __len__(self):
        return self.input_tensor.size(0)

    @classmethod
    def preprocess(cls, src_lang, tgt_lang, reverse=False):
        print("process lines .....")

        src_lines = open(cls.src_dir, 'r').read().strip().split('\n')
        tgt_lines = open(cls.tgt_dir, 'r').read().strip().split('\n')

        src_pairs = [[normalizeString(s) for s in l.split('\t')]
                     for l in src_lines]
        tgt_pairs = [[normalizeString(s) for s in l.split('\t')]
                     for l in tgt_lines]
        pairs = [src_pairs, tgt_pairs]

        _src_lang, _tgt_lang = src_lang[1:], tgt_lang[1:]
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(tgt_lang)
            output_lang = Lang(src_lang)
        else:
            input_lang = Lang(_src_lang)
            output_lang = Lang(_tgt_lang)

        return input_lang, output_lang, pairs

    def indexesFromSentence(self, lang: Lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, lang: Lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device='cpu').view(-1, 1)

    def __getitem__(self):
        input_lang, output_lang, pairs = self.preprocess(
            self.langs[0], self.langs[1], reverse=False)
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])

        self.input_tensor = self.tensorFromSentence(input_lang, pairs[0])
        self.target_tensor = self.tensorFromSentence(output_lang, pairs[1])
        return {'input': self.input_tensor, 'target': self.target_tensor}


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

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
