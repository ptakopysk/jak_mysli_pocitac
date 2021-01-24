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
EPSILON = 0.1
MIN_SIM = 0.3

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

words = list(embeddings.keys())


import random

def cos(u, v):
    product = 0
    absu = 0
    absv = 0
    for x, y in zip(u, v):
        product += x*y
        absu += x*x
        absv += y*y
    return product / (absu**0.5 * absv**0.5)


def random_triple():
    while True:
        prompt = random.choice(words)
        one = random.choice(words)
        two = random.choice(words)
        
        if prompt == one or prompt == two or one == two:
            continue
        
        sim1 = cos(embeddings[prompt], embeddings[one])
        sim2 = cos(embeddings[prompt], embeddings[two])
        if sim1 < MIN_SIM or sim2 < MIN_SIM:
            continue

        if abs(sim1-sim2) > EPSILON:
            return prompt, one, two, sim1, sim2


print('')
print('Vítám tě ve hře Jak myslí počítač! Vždycky ti ukážu slovo, a ty budeš mít za úkol odhadnout, které z dalších dvou slov je podle počítače (podle slovních embedinků FastText) podobnější zadanému slovu.')
print('Odpovídej 1 nebo 2, hru ukončíš 0.')
print()

total = 0
correct = 0

while True:
    prompt, one, two, sim1, sim2 = random_triple()
    
    print()
    print(f'Které slovo je podobnější slovu: {prompt}')
    print(f'1: {one}')
    print(f'2: {two}')
    print(f'0: KONEC HRY')
    answer = input()
    if answer == '0':
        print()
        print('Díky za hru!')
        print(f'Celkové skóre: {correct} správně z {total}, to je {100*correct/total}%')
        break
    elif answer not in ('1', '2'):
        print('Zadej 1 nebo 2!')
    else:
        assert answer in ('1', '2')
        correct_answer = '1' if sim1 > sim2 else '2'
        total += 1
        if answer == correct_answer:
            correct += 1
            print('Správně!')
        else:
            print('Špatně!')
        print(f'cos({prompt}, {one}) = {100*sim1}%')
        print(f'cos({prompt}, {two}) = {100*sim2}%')





