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

N=300

import random
def rand_emb(N=300):
    emb=list()
    for _ in range(N):
        emb.append(random.uniform(-1, 1))
    return emb

lines = list()
for row in range(5):
    for col in range(6):
        x = col*3
        y = row*3
        # svislý čáry
        lines.append( ([x, x], [y, y+2]) )
        lines.append( ([x+1, x+1], [y, y+2]) )
        lines.append( ([x+2, x+2], [y, y+2]) )
        # vodorovný čáry
        lines.append( ([x, x+2], [y, y]) )
        lines.append( ([x, x+2], [y+1, y+1]) )
        lines.append( ([x, x+2], [y+2, y+2]) )
        # čáry křížem
        lines.append( ([x, x+2], [y, y+2]) )
        lines.append( ([x, x+2], [y+2, y]) )

fig, ax = plt.subplots(1, 1)

# no borders
ax.set_frame_on(False)    
# no visible axes
ax.set_xticks([])
ax.set_yticks([])

# from blue to red
emb = rand_emb(300)
for dim, (xs, ys) in zip (emb, lines):
    color = 'b' if dim < 0 else 'r'
    ax.plot(xs, ys, color, alpha=abs(dim))

#ax.plot([1, 3], [1, 3], 'r')
#ax.plot([1, 1], [1, 3], 'b', alpha=1)
#ax.plot([2, 2], [1, 3], 'b', alpha=0.5)
#ax.plot([3, 3], [1, 3], 'b', alpha=0.1)


plt.show()

        

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

