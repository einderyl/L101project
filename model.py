# -*- coding: utf-8 -*-

import nltk
nltk.download('stopwords')

import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation

from nltk import wordpunct_tokenize
from nltk.corpus import stopwords

sw = stopwords.words("english") 
sw += [p for p in punctuation]

def load_data(dataset):
  df = pd.read_csv(f'drive/MyDrive/{dataset}_classed.csv')
  with open(f'drive/MyDrive/{dataset}_bert.pickle', 'rb') as f:
    bert = pickle.load(f)
  with open(f'drive/MyDrive/{dataset}_spacy.pickle', 'rb') as f:
    poses, deps, head_poses = pickle.load(f)
  return df, bert, poses, deps, head_poses

df, bert, poses, deps, head_poses = load_data('cs')

stopwords_vectorizer = TfidfVectorizer(vocabulary=sw, tokenizer=wordpunct_tokenize)
stopwords_vectorizer = stopwords_vectorizer.fit(list(df['selftext']))

norm = Normalizer().fit([[0], [1]])

features = [0,0,0,0,0]
for i, feature in enumerate(['dep', 'pos', 'head_pos']):
    features[i] = norm.transform(list(df[feature]))
 
features[3] = bert
features[4] = stopwords_vectorizer.transform(list(df['selftext'])).todense()

X = np.concatenate(features, axis=1)
y = np.array(df['class'])
class_dist = {class_name: len(df[df['class'] == class_name]) for class_name in set(y)}
class_weights = {class_name: class_dist[0] / class_dist[class_name] for class_name in class_dist.keys()}

def evaluate(true_classes, predicts):
  fs = [0 for _ in range(7)]

  for i in range(1,8):
      tp = 0
      fp = 0
      fn = 0

      for true, predicted in zip(true_classes, predicts):
          if true >= i and predicted >= i:
            tp += 1
          elif true >= i and predicted < i:
            fn += 1
          elif true < i and predicted >= i:
            fp += 1

      f = tp / (tp + 0.5*(fp+fn))
      fs[i-1] = f
  print(fs)
  return sum(fs) / 7

scores = []

for train_index, test_index in StratifiedKFold(n_splits=5).split(X, y):
  X_train, y_train = X[train_index], y[train_index]
  X_test, y_test = X[test_index], y[test_index]

  model = RandomForestClassifier(class_weight=class_weights)
  model.fit(X_train, y_train)
  predicts = model.predict(X_test)
  scores.append(evaluate(list(y_test), predicts))
print(np.mean(scores))
