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

def any_bigger(thing, value = 0.25):
    #print(thing)
    #print(thing[0]>value)
    return (thing[0] > value).any()

def all_bigger(thing, value = 0.25):
    a = []
    #print(thing)
    for i in xrange(len(thing)):
        if thing[i] > value:
            a.append(i)
    return a

#k = [['ha', 'ha', 'ha'], ['i', 'will', 'fail', 'this']]
#k2 = get_positions(k)
#print(k2)
#k3 = enumerate_algs(k2[0], k2[1])
#print(k3)

#/data/mifs_scratch/js2166/bpe/data/from_adria/data/dev.alignments /data/mifs_scratch/js2166/bpe/data/from_adria/data/dev.tok.src /data/mifs_scratch/js2166/bpe/data/from_adria/data/dev.tok.trg /data/mifs_scratch/js2166/bpe/data/src.wmap /data/mifs_scratch/js2166/bpe/data/bpe_ids_ja /data/mifs_scratch/js2166/bpe/data/trg.wmap /data/mifs_scratch/js2166/bpe/data/bpe_ids_en /data/mifs_scratch/js2166/bpe/train_added_align_001/dev.atts.out2 -1


#wrd_in = /data/mifs_scratch/js2166/bpe/data/src.wmap
#bpe_in = /data/mifs_scratch/js2166/bpe/data/bpe_ids_ja
#wrd_out = /data/mifs_scratch/js2166/bpe/data/src.wmap
#bpe_out = /data/mifs_scratch/js2166/bpe/data/bpe_ids_en

def run(target_algs, text_in, text_out, attention_pickle, value1=0.2, value2=0.25):
    wrd_in = '/data/mifs_scratch/js2166/bpe/data/src.wmap'
    bpe_in = '/data/mifs_scratch/js2166/bpe/data/bpe_ids_ja'
    wrd_out = '/data/mifs_scratch/js2166/bpe/data/trg.wmap'
    bpe_out = '/data/mifs_scratch/js2166/bpe/data/bpe_ids_en'

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
            #print(sentence_attention[i])
            if any_bigger(sentence_attention[i], value1):
                obtained.append((produce_hard_algn(sentence_attention[i]), i))
            #print(sentence_attention[i])
            #for link in all_bigger(sentence_attention[i][0][0]):
            #    obtained.append((link, i))
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
        #out.write(str(1-float(n)/len(obtained)))
        #out.write('\n')

    #print('\n')
    #print('\n')
    aer1 = 1-float(total_good)/total_total
    aer2 = 1-2*float(total_good)/total_total_total
    out.write('\n')
    out.write('\n')
    out.write(str(aer1))
    out.write(str(aer2))
    out.write('\n')
    out.write('\n')

    ta.close()
    ti.close()
    to.close()
    ap.close()
    out.close()
    os.remove('tmp.pkl')
    return aer1, aer2

if __name__ == "__main__":
    if sys.argv[1] == 'dev':
        target_algs = '/data/mifs_scratch/js2166/bpe/data/from_adria/data/dev.alignments'
        text_in = '/data/mifs_scratch/js2166/bpe/data/from_adria/data/dev.tok.src'
        text_out = '/data/mifs_scratch/js2166/bpe/data/from_adria/data/dev.tok.trg'
        attention_pickle = sys.argv[2]
    elif sys.argv[1] == 'test':
        target_algs = '/data/smt2016/ad465/data/for-felix.jpn-eng-wat/data/test.alignments'
        text_in = '/data/mifs_scratch/js2166/bpe/data/from_adria/data/test.tok.src'
        text_out = '/data/mifs_scratch/js2166/bpe/data/from_adria/data/test.tok.trg'
        attention_pickle = sys.argv[2]
    else:
        target_algs = sys.argv[1]
        text_in = sys.argv[2]
        text_out = sys.argv[3]
        attention_pickle = sys.argv[4]
    aer1, aer2 = run(target_algs = target_algs, text_in = text_in, text_out=text_out, attention_pickle=attention_pickle)
    print("\n\n")
    print(aer1)
    print(aer2)
