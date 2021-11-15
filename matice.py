#!/usr/bin/env python3
#coding: utf-8

import sys

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

N = 17



def draw_word(word, emb, ax):
    # push embeddings into a square matrix
    data = np.reshape(emb[:N**2], (N, N))
    # text label of x axis
    ax.set_xlabel(word)
    # from blue to red
    ax.imshow(data, cmap='seismic')

def rand_emb():
    emb=list()
    for _ in range(N**2):
        emb.append(random.uniform(-5, 5))
    return emb

import random

text = [ ['Pes', 'jitrničku', 'sežral', 'docela', 'maličkou'],
        ['kuchař', 'ho', 'přitom', 'popad', 'praštil', 'ho', 'paličkou'] ]

rows = len(text)
cols = max([len(line) for line in text])

fig, axs = plt.subplots(rows, cols)

# set style for all axes
for row in axs:
    for ax in row:
        # no borders
        ax.set_frame_on(False)    
        # no visible axes
        ax.set_xticks([])
        ax.set_yticks([])

# draw words
for line, row in zip(text, axs):
    for word, ax in zip(line, row):
        draw_word(word, rand_emb(), ax)
        

# make a figure + axes
# Number of rows/columns of the subplot grid.
# ax can be either a single Axes object or an array of Axes objects if more
# than one subplot was created.
#fig, ax = plt.subplots(1, 1, tight_layout=True)
#fig, ax = plt.subplots(1, 1)


#ax.set_xlabel('Pes')
#ax.set_frame_on(False)
#ax.set_xticks([])
#ax.set_yticks([])

# make color map
#my_cmap = matplotlib.colors.ListedColormap(['r', 'w', 'b'])
# set the 'bad' values (nan) to be white and transparent
#my_cmap.set_bad(color='w', alpha=0)
# draw the grid
#for x in range(N + 1):
#    ax.axhline(x, lw=2, color='k', zorder=5)
#    ax.axvline(x, lw=2, color='k', zorder=5)

# draw the boxes
#ax.imshow(data, interpolation='none', cmap=my_cmap, extent=[0, N, 0, N], zorder=0)
#ax.imshow(data, cmap='seismic')

# turn off the axis labels
#ax.axis('off')

plt.show()
