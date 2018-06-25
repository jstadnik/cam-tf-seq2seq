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
    return ans

def enumerate_algs(list1, list2):
    ans = []
    for id1 in list1:
        for id2 in list2:
            ans.append((id1, id2))
    return ans

def word2sub(target_aligns, sentence_in, sentence_out, wmap_d_in, wmap_d_out):
    """Produces """

    target_aligns = target_aligns.strip().split()

    sentence_bpe_in = []
    sentence_bpe_out = []

    unk = '0'

    for word in sentence_in:
        bpe = wmap_d_in.get(word, unk)
        sentence_bpe_in.append(bpe.strip().split())

    for word in sentence_out:
        bpe = wmap_d_out.get(word, unk)
        sentence_bpe_out.append(bpe.strip().split())

    print(sentence_bpe_in)
    print(sentence_bpe_out)


    bpe_in_pos = get_positions(sentence_bpe_in)
    bpe_out_pos = get_positions(sentence_bpe_out)

    bpe_aligns = []

    for i in xrange(len(target_aligns)/2):
        #print i
        bpe_aligns += enumerate_algs(bpe_in_pos[int(target_aligns[2*i])], bpe_out_pos[int(target_aligns[2*i+1])])


    return bpe_aligns, len(sentence_bpe_out), len(sentence_bpe_in)



target_algs = sys.argv[1]
text_in = sys.argv[2]
text_out = sys.argv[3]
wmap_in = sys.argv[4]
wmap_out = sys.argv[5]


d_in = load_wmap(wmap_in)
d_out = load_wmap(wmap_out)

ta = open(target_algs, 'r')
ti = open(text_in, 'r')
to = open(text_out, 'r')


out = open('alignment.bpe', 'w+')

for line in ta:
    sentence_in = ti.readline().strip().split()
    sentence_out = to.readline().strip().split()
    subword_aligns, len1, len2 = word2sub(line, sentence_in, sentence_out, d_in, d_out)
    #leng = len(sentence_out)
    print(subword_aligns)
    align_prob = np.zeros([len1+1, len2+1])
    for x, y in subword_aligns:
        align_prob[x, y] += 1
    align_prob = (align_prob.T/np.sum(align_prob.T, 0)).T
    print(align_prob)


ta.close()
ti.close()
to.close()
out.close()
