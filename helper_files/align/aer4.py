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
    #print(sentence_bpe_out)

    bpe_in_pos = get_positions(sentence_bpe_in)
    bpe_out_pos = get_positions(sentence_bpe_out)

    #print(bpe_in_pos)
    #print(bpe_out_pos)
    #print(target_aligns)

    bpe_aligns = []

    for i in xrange(len(target_aligns)/2):
        #print i
        bpe_aligns += enumerate_algs(bpe_in_pos[int(target_aligns[2*i])], bpe_out_pos[int(target_aligns[2*i+1])])


    return bpe_aligns

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
attention_pickle = sys.argv[8]
if sys.argv[9]:
    num = sys.argv[9]
else:
    num = None


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
shutil.copyfile(attention_pickle, 'tmp.pkl')
ap = open('tmp.pkl', 'rb')

poof = 0
iters = -1
if num:
    iters = int(num)
    #print('\n\n\n'+str(num)+'\n\n\n')

total_good = 0
total_total = 0
total_total_total = 0

open('calculated_aer', 'w').close()
out = open('calculated_aer', 'w+')

for line in ta:
    sentence_in = ti.readline().strip().split()
    sentence_out = to.readline().strip().split()
    subword_aligns = word2sub(line, sentence_in, sentence_out, d_in, d_out)
    sentence_attention = pickle.load(ap)
    obtained = []
    #print(poof)
    #print(sentence_in)
    #print(sentence_out)
    #print(len(sentence_attention))
    for i in xrange(np.max(subword_aligns, 0)[1]+1):
        obtained.append((produce_hard_algn(sentence_attention[i]), i))
    #print(str(subword_aligns))
    #print('\n')
    out.write(str(subword_aligns))
    out.write('\n')
    out.write(str(obtained))
    out.write('\n')
    n = 0
    for algn in obtained:
        if algn in subword_aligns:
            n+=1
    total_good += n
    total_total += len(obtained)
    total_total_total += len(obtained) + len(subword_aligns)
    out.write(str(1-float(n)/len(obtained)))
    out.write('\n')
    poof+=1
    if poof==iters:
        break

print('\n')
print('\n')
print(1-float(total_good)/total_total)
print(1-2*float(total_good)/total_total_total)
out.write('\n')
out.write('\n')
out.write(str(1-float(total_good)/total_total))
out.write(str(1-2*float(total_good)/total_total_total))
out.write('\n')
out.write('\n')

ta.close()
ti.close()
to.close()
ap.close()
out.close()
os.remove('tmp.pkl')
