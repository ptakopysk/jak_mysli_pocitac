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
ap.add_argument('--LIMIT', type=int, help='read this many embeddings', default=100000)
ap.add_argument('--MAXLINES', type=int, help='read this many text lines', default=10)
ap.add_argument('--TXTFILE', help='read text from this file', default="kava.txt")
ap.add_argument('--N', type=int,help='side of square (so a square fits N^2 dimensions)', default=17)
ap.add_argument('--SORT', help='show dimensions sorted', default=False, action=argparse.BooleanOptionalAction)
ap.add_argument('--DRAWLINES', help='represent by lines', default=False, action=argparse.BooleanOptionalAction)
ap.add_argument('--EXP_FOR_OPACITY', type=float, help='exponentiate dimensions to make them more visible', default=1.0)
ap.add_argument('--BESTLINES', type=int, help='only draw this many highest scoring lines', default=0)
ap.add_argument('--BESTLINES_THRESHOLD', type=float, help='draw only lines with score above threshold', default=0.1)
ap.add_argument('--LEFTRIGHT',
    help='represent by lines going -left and +right; for + it goes up/right, for - down/left',
    default=False, action='store_true')
ap.add_argument('--colors', type=int, help='how many colors', default=6)
ap.add_argument('--show', help='show plot', default=True, action=argparse.BooleanOptionalAction)
ap.add_argument('--store', help='store plot', default=True, action=argparse.BooleanOptionalAction)
ap.add_argument('--format', help='format of the stored file (pdf, svg, eps, png...)', default="svg")

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

EMPTY_EMB = [0 for _ in range(embeddings_dim)]

# NUM_OF_DRAWLINES = 26
NUM_OF_DRAWLINES = 18

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
            #drawlines.append( ([x+0.25, x+0.25], [y, y-2]) )
            drawlines.append( ([x+0.50, x+0.50], [y, y-2]) )
            #drawlines.append( ([x+0.75, x+0.75], [y, y-2]) )
            drawlines.append( ([x+1, x+1], [y, y-2]) )
            #drawlines.append( ([x+1.25, x+1.25], [y, y-2]) )
            drawlines.append( ([x+1.50, x+1.50], [y, y-2]) )
            #drawlines.append( ([x+1.75, x+1.75], [y, y-2]) )
            # vodorovný čáry
            drawlines.append( ([x, x+2], [y, y]) )
            #drawlines.append( ([x, x+2], [y-0.25, y-0.25]) )
            drawlines.append( ([x, x+2], [y-0.50, y-0.50]) )
            #drawlines.append( ([x, x+2], [y-0.75, y-0.75]) )
            drawlines.append( ([x, x+2], [y-1, y-1]) )
            #drawlines.append( ([x, x+2], [y-1.25, y-1.25]) )
            drawlines.append( ([x, x+2], [y-1.50, y-1.50]) )
            #drawlines.append( ([x, x+2], [y-1.75, y-1.75]) )
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

def exp_sym(number, exponent):
    if number > 0:
        return number**exponent
    else:
        return -( (-number)**exponent )

COLORS = ('b', 'g', 'r', 'c', 'm', 'y')
STARTS = (
        (0, 0),
        (0.25, 0.25),
        (-0.25, 0.25),
        (0.25, -0.25),
        (-0.25, -0.25),
        (0, 0.25),
        (0, -0.25),
        (0.25, 0),
        (-0.25, 0),
        )

def draw_word(word, emb, ax):    
    # text label of x axis
    ax.set_xlabel(word)
    if args.DRAWLINES:
        if args.BESTLINES:
            # show frame
            ax.set_frame_on(True)
            ax.spines['top'].set_alpha(0)
            ax.spines['right'].set_alpha(0)
            ax.spines['bottom'].set_alpha(0.2)
            #ax.spines['left'].set_color('black')
            ax.spines['left'].set_alpha(0.2)
            #ax.set_xlim(-0.5, drawlines_cols*2+0.5)
            #ax.set_ylim(-drawlines_rows*2-0.5, 0.5)
            ax.set_xlim(-0.1, drawlines_cols*2 + 0.1)
            ax.set_ylim(-drawlines_rows*2 - 0.1, 0.1)
            #plt.xticks(range(drawlines_cols*2+1), [])
            #plt.yticks(range(-drawlines_rows*2-1, 1), [])
            # show points
            for x in range(0, drawlines_cols*2+1, 2):
                for y in range(0, drawlines_rows*2+1, 2):
                    ax.plot(x, -y, 'k,')
            # sort by abs value
            pairs = enumerate(emb)
            pairs_sorted = sorted(pairs, key=lambda pair: -abs(pair[1]))
            for idx, dim in pairs_sorted[:args.BESTLINES]:
                if abs(dim) > args.BESTLINES_THRESHOLD:
                    xs, ys = drawlines[idx]
                    color = 'b' if dim < 0 else 'r'
                    ax.plot(xs, ys, color, alpha=absmax1(dim)**args.EXP_FOR_OPACITY)
        else:
            for dim, (xs, ys) in zip (emb, drawlines):
                color = 'b' if dim < 0 else 'r'
                ax.plot(xs, ys, color, alpha=absmax1(dim)**args.EXP_FOR_OPACITY)
    elif args.LEFTRIGHT:
        # frame
        ax.axis('scaled')
        ax.plot(0, 0, 'k+', alpha=0.2)
        for xs, ys in (
                # axes
                ((-2, 2), (0, 0)),
                ((0, 0), (-2, 2)),
                # central box
                ((-0.5, 0.5), (-0.5, -0.5)),
                ((-0.5, 0.5), (0.5, 0.5)),
                ((-0.5, -0.5), (-0.5, 0.5)),
                ((0.5, 0.5), (-0.5, 0.5))):
            ax.plot(xs, ys, 'k:', alpha=0.2, lw=1)
        # 1: horizontal; 0: vertical
        direction = 0
        color = 0
        x0, y0 = STARTS[color]
        absmax = 0
        for idx, dim in enumerate(emb):
            # move by dim in the direction
            x1 = x0 + dim*direction
            y1 = y0 + dim*(1-direction)
            # draw line
            if args.colors:
                ax.plot((x0, x1), (y0, y1), lw=1, alpha=0.5, color=COLORS[color])
            else:
                ax.plot((x0, x1), (y0, y1), lw=1, alpha=0.5)
            # prepare for next step
            absmax = max(absmax, abs(x1), abs(y1))
            direction = 1 - direction
            x0, y0 = x1, y1
            # may move on to a new color
            if args.colors:
                newcolor = int(idx // (len(emb)/args.colors))
                if newcolor != color:
                    color = newcolor
                    direction = 0
                    x0, y0 = STARTS[color]
        # limit (ensures centering)
        absmax += 0.1
        ax.set_xlim(-absmax, absmax)
        ax.set_ylim(-absmax, absmax)
    else:
        # scale
        emb = [exp_sym(dim, args.EXP_FOR_OPACITY) for dim in emb]
        # clip
        emb = emb[:args.N**2]
        # sort
        if args.SORT:
            emb = sorted(emb)
        # push embeddings into a square matrix
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
# fig, axs = plt.subplots(rows, cols)

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

# ensure subplots fit nicely
# plt.tight_layout()

if args.store:
    outname = re.sub(r'txt$', args.format, args.TXTFILE)
    logging.info(f'Ukládám obrázek do {outname}')
    plt.savefig(outname)

if args.show:
    logging.info(f'Zobrazuji obrázek')
    plt.show()

