# 데이터과학 group 2
# Encoder(LSTM) 
# Decoder(LSTM + ATTENTION)

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from hparams import hparams

hp = hparams()

device = hp.device

class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hp.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, cell):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, (hidden,cell) = self.lstm(output, (hidden,cell))
        return output, (hidden, cell)

    def initHidden(self, device):
        return torch.zeros(1,1,self.hidden_size,device=device)

class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=0):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hp.hidden_size
        self.embedding = nn.Embedding(output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
 
        output = F.relu(output)
        output, (hidden, cell) = self.lstm(output, (hidden, cell))

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, cell, attn_weights

    def initHidden(self, device):  
        return torch.zeros(1, 1, self.hidden_size, device=device)