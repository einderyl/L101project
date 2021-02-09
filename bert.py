# -*- coding: utf-8 -*-

# Imports
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split

import string
import pickle
import pandas as pd
import numpy as np

from transformers import BertModel, BertTokenizerFast, AdamW, get_linear_schedule_with_warmup
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

weights = 'bert-base-uncased'
class CustomBertTokenizer():
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained(weights)

    def __call__(self, post):
          sentences = sent_tokenize(post)
          short = [s[:510] for s in sentences]
          return self.tokenizer(short, is_split_into_words=False, padding=True, return_offsets_mapping=False)


class BertVectorizer(TransformerMixin):
    def __init__(self):
        self.model = BertModel.from_pretrained(weights).cuda()
        self.tokenizer = CustomBertTokenizer()

    def fit(self, texts, y=None):
        return self

    def transform(self, texts, y=None):
        features = []
        no_done = 0

        for text in texts:
            tokenized = self.tokenizer(text)
            train_texts = torch.tensor(tokenized['input_ids'], dtype=torch.long).cuda()
            train_masks = torch.tensor(tokenized['attention_mask']).cuda()

            with torch.no_grad():
                last_hidden_states = self.model(train_texts, attention_mask=train_masks)

            feature = last_hidden_states[0][:,0,:].cpu().numpy()
            features.append(np.mean(feature, axis=0))

            no_done += 1
            if no_done % 1000 == 0:
                print("{}% done".format(no_done * 100 / len(texts)))

        return np.array(features)

df = pd.read_csv('drive/MyDrive/sg_classed.csv')
df = df[df.notna()['selftext']]
vect = BertVectorizer()
X_train = vect.fit_transform(df['selftext'])

with open('drive/MyDrive/sg_bert.pickle', 'wb') as f:
    pickle.dump(X_train, f)

df = pd.read_csv('drive/MyDrive/uk_classed.csv')
df = df[df.notna()['selftext']]
vect = BertVectorizer()
X_train = vect.fit_transform(df['selftext'])

with open('drive/MyDrive/uk_bert.pickle', 'wb') as f:
    pickle.dump(X_train, f)
