import matplotlib.pyplot as plt
from datetime import datetime
import json
import torch
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
# import string
# from string import digits
import pandas as pd
import seaborn as sns
from collections import Counter

# stop_words = stopwords.words("english")


def new_dir():
    now = datetime.now()
    current_time = now.strftime("%d_%m_%Y_%H_%M_%S")

    nf = current_time + '/'
    return nf


def save_plot(data, name):
    plt.figure()
    plt.plot(data)
    plt.savefig(name)


def dump_hp(nf, hp):
    with open(nf + 'hp.json', 'w') as f:
        json.dump(hp, f)


def save_model(nf, model, model_name):
    torch.save(model.state_dict(), nf + model_name + '.pth')


def correct_spellings(x):
    """correct the missplled words of a given tweet"""
    spell = SpellChecker()
    x = x.split()
    misspelled = spell.unknown(x)
    result = map(lambda word: spell.correction(word) if word in misspelled else word, x)
    return " ".join(result)


def cleaning(x):
    """Apply function to a clean a tweet"""
    x = convert_to_lower(x)

    # remove punctuation
    x = remove_punctuation(x)
    # operator = str.maketrans('', '', string.punctuation)  # ????
    # x = x.translate(operator)
    # x = correct_spellings(x)
    x = word_tokenize(x)
    # x = remove_stopwords(x)
    return x


def convert_to_lower(text):
    # return the reviews after convering then to lowercase
    return text.lower()


def remove_punctuation(text):
    # return the reviews after removing punctuations
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(' +', ' ', text)
    return text

# Not Needed - last 2 lines of remove punctuation does this work 
def remove_stopwords(text):
    # return the reviews after removing the stopwords
    exclude = ['$', '&', '+', ':', ';', '=', '@', '|', '<', '>', '^', '*', '%', '-', '#', '\'', '\n']
    exc = set(exclude)
    # remove_digits = str.maketrans('', '', digits)
    # text = text.translate(remove_digits)
    # return ''.join(ch for ch in text if ch not in (exclude))
    filtered_sentence = [w for w in text if w not in exc]
    return ' '.join(filtered_sentence)


def give_len(a):
    return a.size()[0]


def pad_data(data):
    # Find max length of the mini-batch
    # print('=' * 100)
    # print('=' * 100)
    # print(data)
    # print('=' * 100)
    # print(len(data))
    # print('=' * 100)
    if not len(data) <= 2:
        max_len = max(list(map(give_len, (list(zip(*data))[1]))))
        label_list = torch.tensor(list(zip(*data))[0])
        txt_list = list(zip(*data))[1]
        padded_tensors = torch.stack([torch.cat((txt, torch.tensor([0] * (max_len - give_len(txt))).long())) for txt in txt_list])
    else:
        padded_tensors = data[1]
        label_list = data[0]
    return padded_tensors, label_list


def clean_text(df):
    df['reviews'] = df['reviews'].apply(cleaning)
    return df


def get_class_distribution_loaders(obj, device, isSet=False):
    count_dict = Counter()

    if isSet:
        for (y, _) in obj:
            if len(y.shape) >= 1:
                count_dict.update(y.tolist())
            else:
                count_dict.update([y.item()])
        return count_dict

    for (_, y) in obj:
        if len(y.shape) >= 1:
            count_dict.update(y.tolist())
        else:
            count_dict.update([y.item()])
    return count_dict


def plot_data_distr(loaderslist, initdatalist, nf, device):
    train_count = get_class_distribution_loaders(initdatalist[0], device, isSet=True)
    train_count_new = get_class_distribution_loaders(loaderslist[0], device)
    # val_count = get_class_distribution_loaders(initdatalist[1], device, isSet=True)
    # val_count_new = get_class_distribution_loaders(loaderslist[1], device)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    sns.barplot(x=list(train_count.keys()), y=list(train_count.values()), color="salmon", ax=axes[0]).set_title('Train before')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['left'].set_visible(False)

    sns.barplot(x=list(train_count_new.keys()), y=list(train_count_new.values()), color="salmon", ax=axes[1]).set_title('Train after')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].spines['left'].set_visible(False)

    plt.savefig(nf + 'train_before_after.png')

    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))
    # sns.barplot(x=list(val_count.keys()), y=list(val_count.values()), color="salmon", ax=axes[0]).set_title('Val before')
    # axes[0].spines['top'].set_visible(False)
    # axes[0].spines['right'].set_visible(False)
    # axes[0].spines['bottom'].set_visible(False)
    # axes[0].spines['left'].set_visible(False)

    # sns.barplot(x=list(val_count_new.keys()), y=list(val_count_new.values()), color="salmon", ax=axes[1]).set_title('Val after')
    # axes[1].spines['top'].set_visible(False)
    # axes[1].spines['right'].set_visible(False)
    # axes[1].spines['bottom'].set_visible(False)
    # axes[1].spines['left'].set_visible(False)
    # plt.savefig(nf + 'val_before_after.png')
