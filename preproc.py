# -*- coding: utf-8 -*-

import pandas as pd
import requests
import json
import csv
import time
from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot as plt


def print_dist(df):
    classes = [len(df[df['class'] == i]) for i in range(8)]
    print(classes)

def split(df):
    median = np.median(df['score'])
    print(median)
    split_below = df[df['score'] < median]
    split_above = df[df['score'] >= median]
    return split_below, split_above

def classify(df):
    indexes = [[] for _ in range(8)]

    first_class = df[df['score'] <= 1].index
    indexes[0] = first_class

    rest = df[df['score'] > 1]

    for i in range(1, 7):
        below, above = split(rest)
        indexes[i] = below.index
        rest = above

    indexes[7] = rest.index

    classes = np.array([0 for _ in range(len(df))])
    for i in range(8):
        classes[list(indexes[i])] = i

    df['class'] = classes

    return df

cs = pd.read_csv('drive/MyDrive/cs.csv')

cs = classify(cs)
print_dist(cs)
cs.to_csv('drive/MyDrive/cs_classed.csv')

sg = pd.read_csv('drive/MyDrive/sg_classed.csv')

sg = classify(sg)
print_dist(sg)
sg.to_csv('drive/MyDrive/sg_classed.csv')


def plot_dates(df):
    dates = [datetime.fromtimestamp(timestamp).hour for timestamp in df['created_utc']]
    classes = df['class']

    dist = [0 for _ in range(8)]
    for c in range(8):
        hours = [dates[i] for i in range(len(classes)) if classes[i] == c]
        counts = [0 for _ in range(24)]
        for hour in range(24):
            counts[hour] = sum([1 for h in hours if h == hour])
        dist[c] = counts

    for label in range(8):
        plt.plot(dist[label], label=f'Level {label}')
    plt.legend()
    plt.xlabel('Hour of posting')
    plt.ylabel('Number of posts')
    plt.show()

plot_dates(sg)
plot_dates(uk)
