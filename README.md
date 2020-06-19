# pytorch_practice


### Contents
1. XOR
2. MNIST DATA (DNN)
3. CIFAR (CNN - RESNET)
4. vanilla word2vec
5. Advanced word2vec (Hierarchical Softmax, Negative Sampling, Subsampling)
6. FastText
7. CNN - Sentimental Analysis
8. RNN + Attention - NMT(Natural 
<br>

---

# 1. XOR

<br>
<br>

### About XOR
- XOR(Exclusive OR) is True when "A, not B" or "B, not A"
- XOR had not been solved until MLP(Multi-Layer Perceptron) was recogized to solve XOR problems

![XOR](https://github.com/lmhljhlmhljh/pytorch_practice/blob/master/assets/Xor.png)

---

# 8. RNN(LSTM) + Attention

<br>
<br>

## Translation Model
- NMT (Neural Machine Translation)
- French2English translation

## Model (LSTM+ATTENTION)
- Encoder: LSTM
- Decoder: LSTM + Attention

![LSTM+Attention](https://github.com/lmhljhlmhljh/pytorch_practice/blob/master/assets/rnn_attention.png)

## requirement
torch

## How to train
``` bash
python main.py train [dir]
python main.py train result/
```
- [dir] : where to save the checkpoints
- prepare empty directory to save encoder and decoder


## How to evaluate the checkpoints
``` bash
python main.py evaluate [dir]
python main.py evaluate result/
```
- [dir] : checkpoints to be evaluated with bleu score
- ABS_log.json will be made. (key:value - chkpt_num : Average Blue Score)
- pretrained models are in "./result/"

## Calculating Bleu Score
``` bash
python bleu.py
```

![Bleu Score](https://github.com/lmhljhlmhljh/pytorch_practice/blob/master/assets/bleu_score.png)
