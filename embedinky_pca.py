#!/usr/bin/env python3
#coding: utf-8

import sys
import gzip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import regex as re

from sklearn.decomposition import PCA

import argparse

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    # level=logging.DEBUG)
    level=logging.INFO)

ap = argparse.ArgumentParser()
ap.add_argument('--FILE', help='embeddings file',
        default='cc.cs.300.vec.gz')
ap.add_argument('--LIMIT', type=int, help='read this many embeddings',
        default=100000)
ap.add_argument('--D', type=int, help='number of PCA dimensions to return',
        default=100)

args = ap.parse_args()

embeddings = dict()

logging.info(f'Načítám embedinky ze souboru {args.FILE}')
with gzip.open(args.FILE, 'rt') as infile:
    # header
    header = infile.readline().split()
    embeddings_dim = int(header[1])
    logging.info(f'Soubor obsahuje embedinky dimenze {header[1]} pro {header[0]} slov')
    for line in infile:
        fields = line.split()
        word = fields[0]
        emb = [float(x) for x in fields[1:]]
        embeddings[word] = emb
        if len(embeddings) >= args.LIMIT:
            break
logging.info(f'Načetl jsem {len(embeddings)} embedinků ze souboru {args.FILE}')

logging.info(f'Počítám PCA...')
model = PCA(n_components=args.D)
pcaed = model.fit_transform(list(embeddings.values()))

logging.info(f'Vypisuju výsledek...')
print(args.LIMIT, args.D)
for word, pca in zip(embeddings, pcaed):
    print(word, *pca)

logging.info(f'Hotovo.')
