# 데이터과학 group 2
# train 정의

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from hparams import hparams

hp = hparams()

# start token, end token 지정nvtop
SOS_token = 0
EOS_token = 1

max_len=hp.max_len
device=hp.device

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, 
                                decoder_optimizer, criterion, max_length=max_len, device=None) :

	encoder_hidden = encoder.initHidden(device)
	encoder_cell = encoder.initHidden(device)

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

	loss = 0

	for ei in range(input_length):
		encoder_output, (encoder_hidden, encoder_cell) = encoder(input_tensor[ei], encoder_hidden, encoder_cell)
		encoder_outputs[ei] = encoder_output[0, 0]

	decoder_input = torch.tensor([[SOS_token]], device=device)

	decoder_hidden = encoder_hidden
	decoder_cell = encoder_cell

	teacher_forcing_ratio = 0.5

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	if use_teacher_forcing:
		# Teacher forcing 포함: 목표를 다음 입력으로 전달
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
				decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
			loss += criterion(decoder_output, target_tensor[di])
			decoder_input = target_tensor[di]  # Teacher forcing

	else:
		# Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
		for di in range(target_length):
			decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
				decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.squeeze().detach()  # 입력으로 사용할 부분을 히스토리에서 분리

			loss += criterion(decoder_output, target_tensor[di])
			if decoder_input.item() == EOS_token:
				break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length
	