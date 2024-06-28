# %%
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import time
import math
import collections

from pathlib import Path
from pyvi import ViTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 164
hidden_size = 64

all_data_path = '.data/multi30k'
train_en_path = all_data_path + '/train.en'
train_vi_path = all_data_path + '/train.vi'
val_en_path = all_data_path + '/val.en'
val_vi_path = all_data_path + '/val.vi'
test_en_path = all_data_path + '/test2016.en'
test_vi_path = all_data_path + '/test2016.vi'


# %%
SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1, "UNK": 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 2  # Count SOS and EOS

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

# %%
def readLangs(lang1_path, lang2_path):
    print("Reading lines...")

    # Read the file and split into lines
    lines1 = open(lang1_path, encoding='utf-8').\
        read().strip().split('\n')

    lines2 = open(lang2_path, encoding='utf-8').\
        read().strip().split('\n')

    lines1 = [s.lower() for s in lines1]
    lines2 = [ViTokenizer.tokenize(s.lower()) for s in lines2]

    pairs = paired_sentences = list(zip(lines1, lines2))

    # Reverse pairs, make Lang instances

    input_lang = Lang(lang1_path)
    output_lang = Lang(lang2_path)

    return input_lang, output_lang, pairs

# %%
def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

_1, _2, pairs = prepareData(val_en_path, val_vi_path)

# %%
MAX_LENGTH = 80

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden[0]

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden[0]

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.LSTM(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden[0], attn_weights

def indexesFromSentence(lang, sentence):
    res = []
    for word in (sentence.split(' ')):
        if (word not in lang.word2index): 
            word = "UNK"
        res.append(lang.word2index[word])
    return res

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(path1, path2):
    input_lang, output_lang, pairs = prepareData(path1, path2)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

input_lang, output_lang, train_dataloader = get_dataloader(train_en_path, train_vi_path)
_1, _2, val_dataloader = get_dataloader(val_en_path, val_vi_path)
_1, _2, test_dataloader = get_dataloader(test_en_path, test_vi_path)

# %%
def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):

    total_loss = 0
    for i, data in enumerate(dataloader):
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        torch.cuda.empty_cache()

        if (i%100 == 0):
            print('step :', round((i / len(dataloader)) * 100, 2), '% , loss :', loss.item())

    return total_loss / len(dataloader)

def get_bleu(pred_seq, label_seq, k = 4):
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))


    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1

        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score



def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

def change_tensor2word(tensor, lang):

    sentence = []
    for num in tensor:
        if num.item() == EOS_token:
            sentence.append('EOS')
            break
        sentence.append(lang.index2word[num.item()])
    
    return ' '.join(sentence)

def valid_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, input_lang, output_lang):
    
    total_loss, total_bleu = 0, []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            input_tensor, target_tensor = data


            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            
            #print(data)

            for j in range(0, len(input_tensor)):
                
                p0 = input_tensor[j]
                p1 = target_tensor[j]

                src_sen = change_tensor2word(p0, input_lang)
                trg_sen = change_tensor2word(p1, output_lang)

                #print(src_sen + "lmaolmao")
                #print(trg_sen + "bruhbruh")

                output_words, _ = evaluate(encoder, decoder, src_sen, input_lang, output_lang)
                output_sentence = ' '.join(output_words)
                
                #print(output_words)

                bleu = get_bleu(output_sentence, trg_sen)
                total_bleu.append(bleu)

            total_loss += loss.item()
            torch.cuda.empty_cache()

    batch_bleu = sum(total_bleu) / len(total_bleu)
    return total_loss / len(dataloader), batch_bleu

# %%

def save_model(name):
    MODEL_PATH = Path('models')
    MODEL_ENC_NAME = Path(name + '_encoder.pth')
    MODEL_DEC_NAME = Path(name + '_decoder.pth')
    ENC_SAVE_PATH = MODEL_PATH / MODEL_ENC_NAME
    DEC_SAVE_PATH = MODEL_PATH / MODEL_DEC_NAME
    #print(f'Saving model to : {MODEL_SAVE_PATH}')
    torch.save(obj = encoder.state_dict(),
            f = ENC_SAVE_PATH)
    torch.save(obj = decoder.state_dict(),
            f = DEC_SAVE_PATH)

inf = float('inf')
def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=0.001,
               best_loss = inf, best_bleu = 0, input_lang=input_lang, output_lang=output_lang):
    start = time.time()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(0, n_epochs):
        start_time = time.time()
        train_loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        valid_loss, bleu = valid_epoch(val_dataloader, encoder, decoder, 
                                       encoder_optimizer, decoder_optimizer, criterion,
                                       input_lang, output_lang)
        end_time = time.time()

        if valid_loss < best_loss:
            best_loss = valid_loss
            save_model('lstm_best_loss')
        
        if (bleu > best_bleu):
            best_bleu = bleu
            save_model('lstm_best_bleu')
        
        print(f'Epoch: {epoch + 1}')
        print(f'Epoch time : {(end_time - start_time)/3600}hrs')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')
    
    end = time.time()
    print(f'Training time : {(end - start)/3600}hrs')

    test_loss, test_bleu = valid_epoch(test_dataloader, encoder, decoder, 
                                   encoder_optimizer, decoder_optimizer, criterion,
                                   input_lang, output_lang)
    
    print('Test Result : ------------------------')
    print(f'Loss : {test_loss}')
    print(f'BLEU : {test_bleu}')


# %%
def evaluateRandomly(encoder, decoder, n=20):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

print(f'The model has {count_parameters(encoder) + count_parameters(decoder)} params')

train(train_dataloader, encoder, decoder, 100)

# %%
encoder.eval()
decoder.eval()
evaluateRandomly(encoder, decoder)

# %%



