# 데이터과학 group 2

### How to Run a code #################################
#                                                   ###
# python main.py [train/evaluate] [dir]             ###
# python main.py train result/                      ### 
# python main.py evaluate result/                   ###
#                                                   ###
# [train/evaluate]: Choose the mode                 ###
# [dir]                                             ###
#   - train    : where to save checkpoints          ###
#       - prepare empty directory, train            ###
#   - evaluate : The Checkpoints to be evaluated    ###
#       - ABS (Average Bleu Scores) to json file    ###                      
#######################################################

import torch
import torch.nn as nn
from torch import optim

import os, re, random, time, json, time
import argparse
from os import path

from hparams import hparams 
from dataset import prepareData, tensorsFromPair, tensorFromSentence, loading_test_data
from modules import EncoderRNN, AttnDecoderRNN
from seq2seq import train
from Bleu import Bleu


def main(mode, dir_) :

    # Dataset 준비하기
    print("Preparing Datasets...")
    input_lang, output_lang, pairs, max_len = prepareData('eng', 'fra')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == "train" :
        
        # Hyperparameter laod
        hp = hparams()
        hp.save_log(dir_)

        encoder = EncoderRNN(input_lang.n_words, hp.hidden_size).to(device)
        decoder = AttnDecoderRNN(hp.hidden_size, output_lang.n_words, 
                                    dropout_p=hp.dropout, max_length=max_len).to(device)

        print_loss_total = 0    # print_every 마다 초기화
        print_log_total = 0
        losses = []

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=hp.learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=hp.learning_rate)
        training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs), device=device) 
                                                                                for i in range(hp.n_iters)]
        criterion = nn.NLLLoss()
        losses = dict()

        print("Start Training....")
        start = time.time()

        for iter in range(1, hp.n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            # train - optimizing
            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, 
                           decoder_optimizer, criterion, max_length=max_len, device=device)
            print_loss_total += loss

            # 시간 확인차 만든 코드
            print_log_total += loss

            # epoch 500마다 loss 출력
            log_epoch = 500
            if iter % log_epoch == 0 and iter % hp.save_epoch != 0 :
                print_loss_avg = print_log_total / log_epoch
                dur = (time.time()-start)/60
                log = "epoch: {} |\tTime spent: {:.2f}min |\tloss: {:.2f}".format(iter,dur,print_loss_avg)
                print(log)
                
                print_log_total = 0

            # hparams.py에서 지정한 epoch마다 loss 출력
            if iter % hp.save_epoch == 0 :
                print_loss_avg = print_loss_total / hp.save_epoch
                dur = (time.time()-start)/60
                log = "epoch: {} |\tTime spent: {:.2f}min |\tloss: {:.2f}".format(iter,dur,print_loss_avg) 
                print(log)

                # loss 기록 저장
                losses[iter] = print_loss_avg

                # 5000 epoch마다 저장하기
                save_name = path.join(dir_,"{}_seq2seq.pt".format(iter))
                torch.save((encoder,decoder), save_name)
                print("{}_seq2seq.pt has just been saved.".format(iter))
                
                print_loss_total = 0

        # losses log - json 파일로 출력
        with open(path.join(dir_,"save_loss.json"),"w",encoding="utf-8") as f :
            json.dump(losses,f)


    # 저장된 체크포인트의 Average Bleu Score을 구함. 
    elif mode == "evaluate" :

        ABS_log = dict()

        # 해당 디렉토리의 모든 checkpoint에 대한 Average Bleu Score을 구함.
        models = [chkpt for chkpt in os.listdir(dir_) if "seq2seq.pt" in chkpt]
        models.sort(key = lambda x : int(re.match("(\d+?)_",x).group(1)))

        if not models :
            print("There is checkpoints in the directory. Please change the directory.")
            exit()

        for model_dir in models :
            model_dir = path.join(dir_,model_dir)

            # Tensor load
            print("Loading Model")
            encoder, decoder = torch.load(model_dir, map_location=device)

            # Load test data
            pairs = loading_test_data("eng","fra")

            print("Start evaluating {}....".format(model_dir))
            whole_bleu_score = 0 
            total_freq = 0

            for num in range(len(pairs)) :

                input_sentence = pairs[num][0]
                target_sentence = pairs[num][1]

                if "m" not in target_sentence and "am" not in target_sentence :
                    
                    print("번역할 문장: {}".format(input_sentence))
                    print("정답 문장: {}".format(target_sentence))

                    with torch.no_grad():
                        
                        input_tensor = tensorFromSentence(input_lang, input_sentence, device=device)
                        input_length = input_tensor.size()[0]
                        encoder_hidden = encoder.initHidden(device=device)
                        encoder_cell = encoder.initHidden(device=device)

                        encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)

                        for ei in range(input_length):
                            encoder_output, (encoder_hidden, encoder_cell) = encoder(input_tensor[ei],
                                                                                    encoder_hidden, encoder_cell)
                            encoder_outputs[ei] += encoder_output[0, 0]

                        decoder_input = torch.tensor([[0]], device=device)  # SOS Token을 input으로

                        decoder_hidden = encoder_hidden
                        decoder_cell = encoder_cell

                        decoded_words = []
                        #decoder_attentions = torch.zeros(max_length, max_length)

                        for di in range(max_len):
                            decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
                                                    decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
                            #decoder_attentions[di] = decoder_attention.data

                            topv, topi = decoder_output.data.topk(1)
                            # EOS TOKEN
                            if topi.item() == 1:
                                decoded_words.append('<EOS>')
                                break
                            else:
                                decoded_words.append(output_lang.index2word[topi.item()])

                            decoder_input = topi.squeeze().detach()

                        target = target_sentence.split()

                        # Bleu score 구하기
                        bleu = Bleu(decoded_words,target)
                        bleu_score = bleu.scoring()
                        
                        whole_bleu_score += bleu_score
                        total_freq += 1

                        print("번역된 문장: {}".format(" ".join(decoded_words)))
                        print("Bleu Score: {:.2%}".format(bleu_score))
                        print("############################################")

            Average_Bleu_Score = whole_bleu_score/total_freq

            iter_num = re.search("\d+",model_dir).group()
            print("Bleu Score of {}pt-ABS: {:.2%}".format(iter_num,Average_Bleu_Score))

            ABS_log[iter_num] = Average_Bleu_Score

            time.sleep(2)

        with open(path.join(dir_,"ABS_log.json"),"w") as f :
            json.dump(ABS_log,f)

    else :
        print("Please choose your mode as train/evaluate")

if __name__ == "__main__" :

    # Argparse
    parser = argparse.ArgumentParser(description='choose the options')
    parser.add_argument('mode', help='choose train or evalutate')
    parser.add_argument('dir_',help="""
Mode:train - Path to save the model
Mode:evaluate - Path to load the model""")
    args = parser.parse_args()

    mode = args.mode
    dir_ = args.dir_

    main(mode, dir_)