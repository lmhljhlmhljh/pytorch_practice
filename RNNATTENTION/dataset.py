# 데이터과학 group 2
# 데이터 정제 (load file and make word2idx)

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from hparams import hparams

# start token, end token 지정
SOS_token = 0
EOS_token = 1

hp = hparams()
device = hp.device

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # SOS 와 EOS 포함

    def addSentence(self, sentence):
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

def readLangs(lang1, lang2):

    # 파일을 읽고 줄로 분리
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # 모든 줄을 쌍으로 분리하고 정규화
    pairs = [[s for s in l.split('\t')] for l in lines]

    # 영어와 프랑스어의 순서를 바꿔주기
    pairs = [list(reversed(p)) for p in pairs]
    input_lang = Lang(lang2)
    output_lang = Lang(lang1)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    max_len = getMAXLENGTH(pairs)
    return input_lang, output_lang, pairs, max_len

def getMAXLENGTH(s):
    length=set()
    for input_lang, output_lang in s :
        length.add(len(input_lang.split()))
        length.add(len(output_lang.split()))
    return max(length)+1

def indexesFromSentence(lang, sentence):
    # UNK 처리 완료
    return [lang.word2index[word] if word in lang.word2index else lang.n_words-1 for word in sentence.split(' ')]      

def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], device=device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device=device)
    return (input_tensor, target_tensor)

def loading_test_data(lang1, lang2) :
    lines = open('data/%s-%s_test.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    pairs = [[s for s in l.split('\t')] for l in lines]
    pairs = [list(reversed(p)) for p in pairs]

    return pairs 


if __name__ == "__main__" :
    input_lang, output_lang, pairs, max_len = prepareData('eng', 'fra')
    print(random.choice(pairs))