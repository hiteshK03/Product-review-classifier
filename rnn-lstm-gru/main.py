import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import json
import pickle
from torchtext.vocab import GloVe, FastText, CharNGram
# from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor
# from torchtext.legacy.experimental.datasets.text_classification import TextClassificationDataset
from torch.utils.data import DataLoader

from utils import new_dir, dump_hp, save_model, plot_data_distr, clean_text, cleaning
from models import LSTM, RNN, GRU
from data import give_data, give_iters, Tokenizer, give_embeddings
from train_test import start_train_val, give_class_report


justTest = False
device=torch.device('cuda')


def give_model(name, vocab, hp):
    if name == 'LSTM':
        model = LSTM(len(vocab), hp['embedding_size'], hp['hidden_size'], hp['num_layers'], device=hp['device'], isbidirectional=hp['isbidirectional']).to(hp['device'])
    elif name == 'RNN':
        model = RNN(len(vocab), hp['embedding_size'], hp['hidden_size'], hp['num_layers'], device=hp['device']).to(hp['device'])
    elif name == 'GRU':
        model = GRU(len(vocab), hp['embedding_size'], hp['hidden_size'], hp['num_layers'], device=hp['device'], isbidirectional=hp['isbidirectional']).to(hp['device'])
    else:
        model = None
        raise ValueError("This model is not defined in the Library. Use one of LSTM, RNN or GRU instead")
    return model


def driver(train_file, test_file):
    hp_ = {}

    hp_['hidden_size'] = 512
    hp_['num_layers'] = 3
    hp_['embedding_size'] = 300
    hp_['learning_rate'] = 0.001
    hp_['num_epochs'] = 30
    hp_['batch_size'] = 1024
    hp_['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    hp_['model_name'] = 'LSTM'  # Options: LSTM, RNN, GRU
    hp_['embeddings'] = 'glove'  # Options: glove, fasttext or None
    hp_['isbidirectional'] = True

    train_data, val_data, test_data, vocab = give_data(embeddings=hp_['embeddings'])
    train_iter, val_iter, test_iter = give_iters(hp_['batch_size'], hp_['device'], [('train', train_data), ('val', val_data), ('test', test_data)])

    # experiments = [{'model_name': 'RNN', 'bi': False}, {'model_name': 'GRU', 'bi': False}, {'model_name': 'GRU', 'bi': True}, {'model_name': 'LSTM', 'bi': False}, {'model_name': 'LSTM', 'bi': True}]
    # experiments = [{'model_name': 'GRU', 'bi': False}, {'model_name': 'GRU', 'bi': True}, {'model_name': 'LSTM', 'bi': False}, {'model_name': 'LSTM', 'bi': True}]

    # for exp in experiments:
    hp = hp_.copy()
    # hp['model_name'] = exp['model_name']
    # hp['isbidirectional'] = exp['bi']

    model = give_model(hp['model_name'], vocab, hp)

    if not justTest:
        if hp['embeddings'] is not None:
            pretrained_embeddings = vocab.vectors
            model.embedding.weight.data.copy_(pretrained_embeddings)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=hp['learning_rate'])

    nf = new_dir()
    os.mkdir(nf)
    plot_data_distr([train_iter, val_iter], [train_data, val_data], nf, hp['device'])

    if not justTest:
        model = start_train_val(model, hp['device'], hp['num_epochs'], criterion, optimizer, train_iter, val_iter, nf)

        dump_hp(nf, hp)
        save_model(nf, model, model_name=hp['model_name'])
        give_class_report(model, hp['device'], val_iter=val_iter, nf=nf)

    else:
        model.load_state_dict(torch.load('v2/12_04_2021_19_41_02/LSTM.pt'))
        give_class_report(model, hp['device'], val_iter=val_iter, nf=nf)


def give_preds(text, model_path):
    # tokenizer = Tokenizer()
    # df = pd.DataFrame([[text, 0]], columns=['reviews', 'ratings'])
    # csv_file = clean_text(df)

    # reviews = csv_file['reviews'].values
    # ratings = csv_file['ratings'].values

    # data = [i for i in zip(ratings, reviews)]
    # text_transform = sequential_transforms(tokenizer.tokenize, vocab_func(vocab), totensor(dtype=torch.long))
    # label_transform = sequential_transforms(totensor(dtype=torch.long))
    # transforms = (label_transform, text_transform)
    # dataset = TextClassificationDataset(data, vocab, transforms)
    # loader = DataLoader(dataset, batch_size=1)
    pkl_file = open('vocab.pkl','rb')
    vocab = pickle.load(pkl_file)
    # print(vocab)
    text = cleaning(text)
    indexed = [vocab.stoi[t] for t in text]
    # print(indexed)

    # print(1/0)
    json_file = open('01_05_2021_19_43_15/hp.json')
    hp = json.load(json_file)

    embd = give_embeddings('glove',True,text)
    # print(embd)
    model = give_model('LSTM',vocab,hp)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    data = torch.LongTensor(indexed).to(device)
    data = data.unsqueeze(1)
    # preds = model(data)
    # data = embd.to(device)
    # data = torch.transpose(data, 0, 1).long()
    # print(data.shape)
    val = model(data)
    print(val)
    max_preds = val.argmax(dim = 1)
    print(max_preds.item()+1)
    return max_preds.item()
    # print(val.tolist())
    # return val.tolist()

if __name__ == '__main__':
    # driver('../train.csv','../test.csv')
    text = "This product is okay."
    # direc = '01_05_2021_19_43_15/'
    give_preds(text, '01_05_2021_19_43_15/LSTM.pth')