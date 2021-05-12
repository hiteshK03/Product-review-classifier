import torch
import torchtext
# from torchtext.data import Field, TabularDataset
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchtext.legacy.data import BucketIterator

from torchtext.vocab import GloVe, FastText, CharNGram
# from torchtext.experimental.vocab import vocab
from torchtext.experimental.transforms import TextSequentialTransforms, VocabTransform
from torchtext.experimental.datasets.text_classification import TextClassificationDataset
from torchtext.experimental.functional import sequential_transforms, vocab_func, totensor
from torchtext.data.utils import get_tokenizer
# from torch.nn import Sequential

import pandas as pd
import spacy
import pickle
import numpy as np
from collections import Counter, OrderedDict
from alive_progress import alive_bar

from utils import clean_text, pad_data

spacy_en = spacy.load('en_core_web_sm')


def tokenize(text):
    return [toke.text for toke in spacy_en.tokenizer(text)]


# def give_fields():
#     review = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
#     rating = Field(sequential=False, use_vocab=False)

#     fields = {'reviews': ('review', review), 'ratings': ('rating', rating)}
#     fields_test = {'reviews': ('review', review)}
#     return review, rating, fields, fields_test

def give_embeddings(embd='glove', only_sent=False, data=None):
    # Data can be list of strings. ret: 2D tensor of embeddings
    vec = None
    if embd == 'glove':
        vec = GloVe(name='840B', dim=300)
    elif embd == 'fasttext':
        vec = FastText(language='en')
    else:
        vec = CharNGram()
    if only_sent:
        ret = vec.get_vecs_by_tokens(data, lower_case_backup=True)
        return ret
    return vec

def build_vocab_from_data(data, tokenizer, embeddings, **vocab_kwarg):

    token_freqs = Counter()

    for row in data:
        tokens = tokenizer.tokenize(row)
        token_freqs.update(tokens)

    vocab = torchtext.vocab.Vocab(token_freqs, **vocab_kwarg)
    if embeddings is not None:
        vocab.load_vectors(give_embeddings(embd=embeddings))
    return vocab

def give_data(train_file='../train.csv', val_file='../gold_test.csv', test_file='../test.csv', embeddings='glove'):

    csv_file = pd.read_csv(train_file)
    with alive_bar(1, title='Cleaning data') as bar:
        csv_file = clean_text(csv_file)
        bar()
    reviews = csv_file['reviews'].values

    # counter = Counter()
    # for row in reviews:
    #     counter.update(row)

    # sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    # ordered_dict = OrderedDict(sorted_by_freq_tuples)
    # v1 = vocab(ordered_dict)

    tokenizer = Tokenizer()
    vocab = build_vocab_from_data(reviews, tokenizer, embeddings, max_size=25000)
    output = open('vocab.pkl','wb')
    pickle.dump(vocab, output)
    # print(1/0)

    # print(vocab.vectors)
    # if vocab.vectors is None:
    #     raise ValueError

    print('')
    print('=' * 100)
    print('Unique words in vocab: ' + str(len(vocab)))
    print('10 most common words in vocab: ', vocab.freqs.most_common(10))
    print('=' * 100)
    print('')

    # vocab = build_vocab(reviews, TextSequentialTransforms(get_tokenizer('spacy')))
    # text_transform = Sequential(TextSequentialTransforms(get_tokenizer('spacy')), VocabTransform(v1), ToTensor())
    # label_transform = ToTensor()

    # review, _, fields, fields_test = give_fields()

    # train_data, val_data = TabularDataset.splits(path='.', train=train_file, validation=val_file, format='csv', fields=fields)

    # _, test_data = TabularDataset.splits(path='.', train=test_file, test=test_file, format='csv', fields=fields_test)

    # review.build_vocab(train_data, vectors=embd)
    # input_ = review.vocab

    train_data = data_to_dataset(train_file, tokenizer, vocab)
    val_data = data_to_dataset(val_file, tokenizer, vocab)
    test_data = data_to_dataset(test_file, tokenizer, vocab, isTest=True)

    return train_data, val_data, test_data, vocab


def give_iters(BATCH_SIZE, DEVICE, data_list):
    # return BucketIterator.splits(tuple(data_list), batch_sizes=[BATCH_SIZE] * len(data_list), device=DEVICE, sort=False, sort_key=None)
    iters = []

    for name_data in data_list:
        name = name_data[0]
        data = name_data[1]
        if name == 'train':
            counter = Counter([d[0].item() - 1 for d in data])
            class_sample_count = [counter[i] for i in range(5)]  # dataset has 8 class-1 samples, 13 class-2 samples, etc.
            weights = 1 / torch.Tensor(class_sample_count)
            # print(len(data))
            weights_samples = [weights[d[0] - 1] for d in data]
            sampler = WeightedRandomSampler(weights_samples, num_samples=len(weights_samples))
        else:
            sampler = None
        loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler, collate_fn=pad_data)
        # print(len(loader))
        iters.append(loader)
    return iters


def data_to_dataset(data, tokenizer, vocab, isTest=False):
    csv_file = pd.read_csv(data)
    with alive_bar(1, title='Cleaning data') as bar:
        csv_file = clean_text(csv_file)
        bar()

    reviews = csv_file['reviews'].values
    if not isTest:
        ratings = csv_file['ratings'].values
    else:
        ratings = np.zeros(len(csv_file))

    data = [i for i in zip(ratings, reviews)]
    text_transform = sequential_transforms(tokenizer.tokenize, vocab_func(vocab), totensor(dtype=torch.long))
    label_transform = sequential_transforms(totensor(dtype=torch.long))
    transforms = (label_transform, text_transform)
    dataset = TextClassificationDataset(data, vocab, transforms)

    return dataset

# class TextDataset(Dataset):
#     def __init__(self, csv_file, text_transform, label_transform, isTest=False):
#         self.file = pd.read_csv(csv_file)
#         self.file = clean_text(self.file)
#         self.reviews = self.file['reviews'].values
#         self.text_transform = text_transform
#         self.isTest = isTest
#         if not self.isTest:
#             self.ratings = self.file['ratings'].values
#             self.label_transform = label_transform
#         else:
#             self.ratings = np.zeros(len(self.file))
#             self.label_transform = label_transform

#     def __len__(self):
#         return len(self.file)

#     def __getitem__(self, idx):
#         sample = (self.text_transform(self.reviews[idx]), self.label_transform(self.ratings[idx]) - 1)
#         return sample


class Tokenizer:
    def __init__(self, tokenize_fn='spacy'):

        self.tokenize_fn = torchtext.data.utils.get_tokenizer(tokenize_fn)

    def tokenize(self, s):
        tokens = self.tokenize_fn(s)
        return tokens
