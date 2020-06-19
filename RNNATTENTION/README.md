# Translation Model

## Model (LSTM+ATTENTION)
- Encoder: LSTM
- Decoder: LSTM + Attention

### How to train
`python main.py train [dir]`
`python main.py train result/`
- [dir] : where to save the checkpoints
- prepare empty directory to save encoder and decoder

### How to evaluate the checkpoints
`python main.py evaluate [dir]`
`python main.py evaluate result/`
- [dir] : checkpoints to be evaluated with bleu score
- ABS_log.json will be made. (key:value - chkpt_num : Average Blue Score)

