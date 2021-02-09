# -*- coding: utf-8 -*-

import pickle
import spacy
import pandas as pd
import numpy as np

import nltk
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

pos_tags = (
	"ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", 
	"PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE",
)

pos_ids = { p: i for i, p in enumerate(pos_tags) }

dep_tags = (
	"acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", 
	"auxpass", "case", "cc", "ccomp", "clf", "compound", "conj", "cop", "csubj", 
	"csubjpass", "dative", "dep", "det", "discourse", "dislocated", "dobj", "expl", 
	"fixed", "flat", "goeswith", "intj", "iobj", "list", "mark", "meta", 
	"neg", "nn", "nmod", "npadvmod", "npmod", "nsubj", "nsubjpass", "nummod", "oprd", 
	"obj", "obl", "orphan", "parataxis", "pcomp", "pobj", "poss", "predet", "preconj", "prep", "prt", 
	"punct", "reparandum", "quantmod", "relcl", "ROOT", "vocative", "xcomp", "",   
)

dep_ids = { d: i for i, d in enumerate(dep_tags) }

short_combis = tuple([(p, dep) for p in pos_tags for dep in dep_tags])
short_ids = { c: i for i, c in enumerate(short_combis) }

combis = tuple([(p1, dep, p2) for p1 in pos_tags for dep in dep_tags for p2 in pos_tags])
ids = { c: i for i, c in enumerate(combis) }

def extract(df):
	head_poses = []
	poses = []
	deps = []

	for doc in df['selftext']:
		parsed = nlp(doc)
		pos = [0 for _ in pos_tags]
		dep = [0 for _ in dep_tags]
		head_pos = [0 for _ in pos_tags]
		for i, p in enumerate(parsed):
			pos[pos_ids[p.pos_]] += 1
			dep[dep_ids[p.dep_]] += 1
			head_pos[pos_ids[p.head.pos_]] += 1

		poses.append(pos)
		deps.append(dep)
		head_poses.append(head_pos)

	return poses, deps, head_poses

df = pd.read_csv('drive/MyDrive/uk_classed.csv')
poses, deps, head_poses = extract(df)
with open('drive/MyDrive/uk_spacy.pickle', 'wb') as f:
		pickle.dump([poses, deps, head_poses], f)

df = pd.read_csv('drive/MyDrive/sg_classed.csv')
poses, deps, head_poses = extract(df)
with open('drive/MyDrive/sg_spacy.pickle', 'wb') as f:
		pickle.dump([poses, deps, head_poses], f)
		