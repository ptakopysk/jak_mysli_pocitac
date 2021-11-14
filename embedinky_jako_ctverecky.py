#!/usr/bin/env python3
#coding: utf-8

import sys
import gzip

import logging
logging.basicConfig(
    format='%(asctime)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    # level=logging.DEBUG)
    level=logging.INFO)


FILE='cc.cs.300.vec.gz'
LIMIT=100000

TXTFILE="genesis.txt"

def is_allowed(word):
    return word.isalpha() or word.isnumeric()


embeddings = dict()

logging.info(f'Načítám embedinky ze souboru {FILE}')
with gzip.open(FILE, 'rt') as infile:
    # header
    header = infile.readline().split()
    logging.info(f'Soubor obsahuje embedinky dimenze {header[1]} pro {header[0]} slov')
    for line in infile:
        fields = line.split()
        word = fields[0]
        if is_allowed(word):
            emb = [float(x) for x in fields[1:]]
            embeddings[word] = emb
            if len(embeddings) >= LIMIT:
                break
        else:
            logging.debug(f'Nepovolené slovo: {word}')
logging.info(f'Načetl jsem {len(embeddings)} embedinků ze souboru {FILE}')

import regex as re

with open(TXTFILE) as intext:
    for line in intext:
        words = line.split()
        for word in words:
            cleaned_word = re.sub(r'[^[:alpha:]]', '', word)
            if cleaned_word in embeddings:
                print(embeddings[cleaned_word][:4], word, cleaned_word)
            else:
                print('???', word, cleaned_word)
        print()



