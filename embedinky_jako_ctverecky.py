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
ap.add_argument('--TXTFILE', help='read text from this file', default="kava.txt")
ap.add_argument('--N', help='side of square (so a square fits N^2 dimensions)', default=17)
ap.add_argument('--SORT', help='show dimensions sorted', default=False)
ap.add_argument('--DRAWLINES', help='represent by lines', default=False)

args = ap.parse_args()

embeddings = dict()

embeddings_dim = 0

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

EMPTY_EMB = [0 for _ in range(embeddings_dim)]

NUM_OF_DRAWLINES = 26

import math
if args.DRAWLINES:
    drawlines_cols = int(math.ceil((embeddings_dim/NUM_OF_DRAWLINES)**0.5))
    drawlines_rows = int(math.ceil(embeddings_dim/NUM_OF_DRAWLINES/drawlines_cols))
    logging.info(f'Budu kreslit čárovou reprezentaci o {drawlines_rows} řádcích a {drawlines_cols} sloupcích')

    drawlines = list()
    for row in range(drawlines_rows):
        for col in range(drawlines_cols):
            x = 2*col
            y = -2*row
            # svislý čáry
            drawlines.append( ([x, x], [y, y-2]) )
            drawlines.append( ([x+0.25, x+0.25], [y, y-2]) )
            drawlines.append( ([x+0.50, x+0.50], [y, y-2]) )
            drawlines.append( ([x+0.75, x+0.75], [y, y-2]) )
            drawlines.append( ([x+1, x+1], [y, y-2]) )
            drawlines.append( ([x+1.25, x+1.25], [y, y-2]) )
            drawlines.append( ([x+1.50, x+1.50], [y, y-2]) )
            drawlines.append( ([x+1.75, x+1.75], [y, y-2]) )
            # vodorovný čáry
            drawlines.append( ([x, x+2], [y, y]) )
            drawlines.append( ([x, x+2], [y-0.25, y-0.25]) )
            drawlines.append( ([x, x+2], [y-0.50, y-0.50]) )
            drawlines.append( ([x, x+2], [y-0.75, y-0.75]) )
            drawlines.append( ([x, x+2], [y-1, y-1]) )
            drawlines.append( ([x, x+2], [y-1.25, y-1.25]) )
            drawlines.append( ([x, x+2], [y-1.50, y-1.50]) )
            drawlines.append( ([x, x+2], [y-1.75, y-1.75]) )
            # čáry křížem
            drawlines.append( ([x, x+2], [y, y-2]) )
            drawlines.append( ([x, x+2], [y-2, y]) )
            # čáry napůl křížem
            drawlines.append( ([x, x+2], [y, y-1]) )
            drawlines.append( ([x, x+2], [y-1, y-2]) )
            drawlines.append( ([x, x+2], [y-2, y-1]) )
            drawlines.append( ([x, x+2], [y-1, y]) )
            drawlines.append( ([x, x+1], [y, y-2]) )
            drawlines.append( ([x+1, x+2], [y, y-2]) )
            drawlines.append( ([x, x+1], [y-2, y]) )
            drawlines.append( ([x+1, x+2], [y-2, y]) )

def avg(list):
    return sum(list)/len(list)

def absavg(list):
    return sum([abs(x) for x in list])/len(list)

def absmax1(number):
    a = abs(number)
    return a if a < 1 else 1

def draw_word(word, emb, ax):    
    # text label of x axis
    ax.set_xlabel(word)
    if args.DRAWLINES:
        for dim, (xs, ys) in zip (emb, drawlines):
            color = 'b' if dim < 0 else 'r'
            ax.plot(xs, ys, color, alpha=absmax1(dim))
    else:
        # push embeddings into a square matrix
        emb = emb[:args.N**2]
        if args.SORT:
            emb = sorted(emb)
        data = np.reshape(emb, (args.N, args.N))
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

px = 1/plt.rcParams['figure.dpi']
fig, axs = plt.subplots(rows, cols, figsize=(1920*px, 1080*px))

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
    logging.info(f'Vykresluji řádek: ' + ' '.join(line))
    for word, ax in zip(line, row):
        cleaned_word = re.sub(r'[^[:alpha:]]', '', word)
        if cleaned_word in embeddings:
            emb = embeddings[cleaned_word]
            draw_word(cleaned_word, emb, ax)
            #print(word, 'MAX:', max(emb), 'MIN:', min(emb),
            #        'AVG:', avg(emb), 'ABSAVG:', absavg(emb))
        else:
            draw_word(cleaned_word, EMPTY_EMB, ax)

logging.info(f'Vykresluji obrázek')

outname = re.sub(r'txt$', 'pdf', args.TXTFILE)
logging.info(f'Ukládám obrázek do {outname}')
plt.savefig(outname)

logging.info(f'Zobrazuji obrázek')
plt.show()

