# 8. RNN(LSTM) + Attention

<br>
<br>

### Translation Model
- NMT (Neural Machine Translation)
- French2English translation

### Model (LSTM+ATTENTION)
- Encoder: LSTM
- Decoder: LSTM + Attention

![LSTM+Attention](https://github.com/lmhljhlmhljh/pytorch_practice/blob/master/RNNATTENTION/assets/rnn_attention.png)

### requirement
torch

### How to train
``` bash
python main.py train [dir]
python main.py train result/
```
- [dir] : where to save the checkpoints
- prepare empty directory to save encoder and decoder

### How to evaluate the checkpoints
``` bash
python main.py evaluate [dir]
python main.py evaluate result/
```
- [dir] : checkpoints to be evaluated with bleu score
- ABS_log.json will be made. (key:value - chkpt_num : Average Blue Score)

### Calculating Bleu Score
``` bash
python bleu.py
```

![Bleu Score](https://github.com/lmhljhlmhljh/pytorch_practice/blob/master/RNNATTENTION/assets/bleu_score.png)
