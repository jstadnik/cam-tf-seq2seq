import numpy as np
import sys
import logging
import pickle
import shutil
import os

def load_wmap(path, inverse=False):
    with open(path) as f:
        d = dict(line.strip().split(None, 1) for line in f)
        if inverse:
            d = dict(zip(d.values(), d.keys()))
        for (s, i) in [('<s>', '1'), ('</s>', '2')]:
            if not s in d or d[s] != i:
                logging.warning("%s has not ID %s in word map %s!" % (s, i, path))
        return d

def get_positions(sentence):
    position = 0
    ans = []
    for actual_word in sentence:
        word_pos = []
        for subword in actual_word:
            word_pos.append(position)
            position +=1
        ans.append(word_pos)
    return ans, position

def enumerate_algs(list1, list2):
    ans = []
    for id1 in list1:
        for id2 in list2:
            ans.append((id1, id2))
    return ans

def word2sub(target_aligns, sentence_in, sentence_out, wmap_d_in, wmap_d_out):

    #print(target_aligns)
    #print('\n')


    target_aligns = target_aligns.strip().split()
    #sentence_in = sentence_in.strip().split()
    #sentence_out = sentence_out.strip().split()

    sentence_bpe_in = []
    sentence_bpe_out = []

    unk = '0'

    #print(' '.join([wmap_d_in[w] if (w in wmap_d_in) else unk for w in sentence_in]))

    for word in sentence_in:
        bpe = wmap_d_in.get(word, unk)
        sentence_bpe_in.append(bpe.strip().split())

    for word in sentence_out:
        bpe = wmap_d_out.get(word, unk)
        sentence_bpe_out.append(bpe.strip().split())

    #print(sentence_bpe_in)
    #print('\n')
    #print(sentence_bpe_out)
    #print('\n')

    with open('bpe_out', 'a+') as bpeout:
        for idd in sentence_bpe_out:
            bpeout.write(str(idd[0]))
            bpeout.write(' ')
        bpeout.write("\n")

    bpe_in_pos, len2 = get_positions(sentence_bpe_in)
    bpe_out_pos, len1 = get_positions(sentence_bpe_out)

    #print(bpe_in_pos)
    #print(bpe_out_pos)
    #print(target_aligns)

    bpe_aligns = []

    for i in xrange(len(target_aligns)/2):
        #print i
        bpe_aligns += enumerate_algs(bpe_in_pos[int(target_aligns[2*i])], bpe_out_pos[int(target_aligns[2*i+1])])


    return bpe_aligns, len1, len2

def produce_hard_algn(soft_algn):
    return np.argmax(soft_algn)

#k = [['ha', 'ha', 'ha'], ['i', 'will', 'fail', 'this']]
#k2 = get_positions(k)
#print(k2)
#k3 = enumerate_algs(k2[0], k2[1])
#print(k3)


target_algs = sys.argv[1]
text_in = sys.argv[2]
text_out = sys.argv[3]
wrd_in = sys.argv[4]
bpe_in = sys.argv[5]
wrd_out = sys.argv[6]
bpe_out = sys.argv[7]
iters = int(sys.argv[8])

poof = 0

#d_in = load_wmap(wmap_in)
d_in = {}
with open(wrd_in, 'r') as wrd:
    with open(bpe_in, 'r') as bpe:
        for line1 in wrd:
            line2 = bpe.readline()
            d_in[line1.strip().split()[0]] = line2
d_out = {}
with open(wrd_out, 'r') as wrd:
    with open(bpe_out, 'r') as bpe:
        for line1 in wrd:
            line2 = bpe.readline()
            d_out[line1.strip().split()[0]] = line2


ta = open(target_algs, 'r')
ti = open(text_in, 'r')
to = open(text_out, 'r')


total_good = 0
total_total = 0
total_total_total = 0

open('bpe_aligns', 'wb').close()
out = open('bpe_aligns', 'ab')
open('bpe_out', 'w').close()

for line in ta:
    if poof==iters:
        break
    sentence_in = ti.readline().strip().split()
    sentence_out = to.readline().strip().split()
    #print(len(sentence_in))
    #print('\n')
    #print(len(sentence_out))
    #print('\n')
    subword_aligns, len1, len2 = word2sub(line, sentence_in, sentence_out, d_in, d_out)
    #print(subword_aligns)
    #print('\n')
    #print(len1)
    #print(len2)
    #print('\n')
    align_prob = np.zeros([len1, len2])
    #print(align_prob.shape)
    for x, y in subword_aligns:
        align_prob[y, x] += 1
    #print(align_prob)
    #print('\n')
    #print('\n')
    #print('\n')
    pickle.dump(align_prob, out, pickle.HIGHEST_PROTOCOL)
    poof+=1


ta.close()
ti.close()
to.close()
out.close()
