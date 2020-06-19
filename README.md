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

---
# 1. XOR


---
# 8. RNN(LSTM) + Attention

# Translation Model

## Model (LSTM+ATTENTION)
- Encoder: LSTM
- Decoder: LSTM + Attention

## requirement
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


