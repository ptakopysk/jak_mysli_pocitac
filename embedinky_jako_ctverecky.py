#!/usr/bin/env python3
#coding: utf-8

import sys
import gzip

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import regex as re

import argparse

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    # level=logging.DEBUG)
    level=logging.INFO)

ap = argparse.ArgumentParser()
ap.add_argument('--FILE', help='embeddings file', default='cc.cs.300.vec.gz')
ap.add_argument('--LIMIT', help='read this many embeddings', default=100000)
ap.add_argument('--MAXLINES', help='read this many text lines', default=10)
ap.add_argument('--TXTFILE', help='read text from this file', default="olasce.txt")
ap.add_argument('--N', help='side of square (so a square fits N^2 dimensions)', default=17)

args = ap.parse_args()

EMPTY_EMB = [0 for _ in range(args.N**2)]

embeddings = dict()

logging.info(f'Načítám embedinky ze souboru {args.FILE}')
with gzip.open(args.FILE, 'rt') as infile:
    # header
    header = infile.readline().split()
    logging.info(f'Soubor obsahuje embedinky dimenze {header[1]} pro {header[0]} slov')
    for line in infile:
        fields = line.split()
        word = fields[0]
        emb = [float(x) for x in fields[1:]]
        embeddings[word] = emb
        if len(embeddings) >= args.LIMIT:
            break
logging.info(f'Načetl jsem {len(embeddings)} embedinků ze souboru {args.FILE}')

def avg(list):
    return sum(list)/len(list)

def absavg(list):
    return sum([abs(x) for x in list])/len(list)

def draw_word(word, emb, ax):    
    # push embeddings into a square matrix
    data = np.reshape(emb[:args.N**2], (args.N, args.N))
    # text label of x axis
    ax.set_xlabel(word)
    # from blue to red
    # ax.imshow(data, cmap='seismic', vmin=-1, vmax=1)
    ax.imshow(data, cmap='seismic', vmin=-0.5, vmax=0.5)

logging.info(f'Načítám text ze souboru {args.TXTFILE}')
with open(args.TXTFILE) as intext:
    text = list()
    for line in intext:
        words = line.split()
        text.append(words)
    logging.info(f'Načetl jsem {len(text)} řádek ze souboru {args.TXTFILE}')

text = text[:args.MAXLINES]

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
            emb = embeddings[cleaned_word][:args.N**2]
            draw_word(cleaned_word, emb, ax)
            #print(word, 'MAX:', max(emb), 'MIN:', min(emb),
            #        'AVG:', avg(emb), 'ABSAVG:', absavg(emb))
        else:
            draw_word(cleaned_word, EMPTY_EMB, ax)

logging.info(f'Vykresluji obrázek')

plt.show()

