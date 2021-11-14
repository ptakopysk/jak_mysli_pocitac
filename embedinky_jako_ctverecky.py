#!/usr/bin/env python3
#coding: utf-8

import sys
import gzip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import regex as re

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    # level=logging.DEBUG)
    level=logging.INFO)


FILE='cc.cs.300.vec.gz'
LIMIT=100000

N = 17

TXTFILE="genesis.txt"

# [-1, 0, 0, ..., 0, 0, 1]
EMPTY_EMB = [-1] + [0 for _ in range(N**2-2)] + [1]

embeddings = dict()

logging.info(f'Načítám embedinky ze souboru {FILE}')
with gzip.open(FILE, 'rt') as infile:
    # header
    header = infile.readline().split()
    logging.info(f'Soubor obsahuje embedinky dimenze {header[1]} pro {header[0]} slov')
    for line in infile:
        fields = line.split()
        word = fields[0]
        emb = [float(x) for x in fields[1:]]
        embeddings[word] = emb
        if len(embeddings) >= LIMIT:
            break
logging.info(f'Načetl jsem {len(embeddings)} embedinků ze souboru {FILE}')

def draw_word(word, emb, ax):    
    # push embeddings into a square matrix
    data = np.reshape(emb[:N**2], (N, N))
    # text label of x axis
    ax.set_xlabel(word)
    # from blue to red
    ax.imshow(data, cmap='seismic')

logging.info(f'Načítám text ze souboru {TXTFILE}')
with open(TXTFILE) as intext:
    text = list()
    for line in intext:
        words = line.split()
        text.append(words)
    logging.info(f'Načetl jsem {len(text)} řádek ze souboru {TXTFILE}')

# !!!!
# text = text[:4]

rows = len(text)
cols = max([len(line) for line in text])

logging.info(f'Vytvářím {rows} řádek a {cols} sloupců')

fig, axs = plt.subplots(rows, cols)
    
# set style for all axes
for row in axs:
    for ax in row:
        # no borders
        ax.set_frame_on(False)    
        # no visible axes
        ax.set_xticks([])
        ax.set_yticks([])

logging.info(f'Vykresluji slova')

# draw words
for line, row in zip(text, axs):
    for word, ax in zip(line, row):
        cleaned_word = re.sub(r'[^[:alpha:]]', '', word)
        if cleaned_word in embeddings:
            draw_word(word, embeddings[cleaned_word][:N**2], ax)
        else:
            draw_word(word, EMPTY_EMB, ax)

logging.info(f'Vykresluji obrázek')

plt.show()

