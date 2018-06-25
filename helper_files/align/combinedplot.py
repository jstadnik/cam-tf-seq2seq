import sys, os
import aer6
import aer5
#from aer6 import run
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

target_algs = '/data/mifs_scratch/js2166/bpe/data/from_adria/data/dev.alignments'
text_in = '/data/mifs_scratch/js2166/bpe/data/from_adria/data/dev.tok.src'
text_out = '/data/mifs_scratch/js2166/bpe/data/from_adria/data/dev.tok.trg'
baseline = '/data/mifs_scratch/js2166/bpe/train_baseline_fixed'
directory = sys.argv[1]
#directory2 = sys.argv[1]
maxm = -1

sum_up = os.path.join(directory, 'sum_up')
open(sum_up, 'w').close()



# add an axis and bleu
n = 0
step = []
bleu = []
fname2 = os.path.join(directory, "eugh_short")
if os.path.isfile(fname2):
    with open(fname2, 'r') as f:
        for line in f:
            line = line.strip().split()
            if line[0] == 'global':
                if int(line[2]) != 2000:
                    step.append(int(line[2]))
                    n += 1
            elif line[0][0] == 'n' or line[0][0] == 's':
                bleu.append(float(line[1][:-1]))
                n += 1
            if n == maxm:
                break
else:
    start = int(sys.argv[3])
    step = range(start, start+n*2000, 2000)

nb = 0
stepb = []
bleub = []
fname2 = os.path.join(baseline, "eugh_short")
if os.path.isfile(fname2):
    with open(fname2, 'r') as f:
        for line in f:
            line = line.strip().split()
            if line[0] == 'global':
                if int(line[2]) != 2000:
                    stepb.append(int(line[2]))
                    n += 1
            elif line[0][0] == 'n' or line[0][0] == 's':
                bleub.append(float(line[1][:-1]))
                nb += 1
            if nb == maxm:
                break
else:
    start = int(sys.argv[3])
    step = range(start, start+n*2000, 2000)

plt.plot(step, bleu, 'b-', stepb, bleub, 'g-')
plt.axis([0, 100000, 0, 23])
plt.title("BLEU score curve")
plt.xlabel("step")
plt.ylabel("BLEU")
plt.savefig(os.path.join(directory, 'bleu_bas.png'))
plt.clf()

# Calculate & plot AER6


aers = os.path.join(directory, "aerscores2")
if os.path.isfile(aers):
    with open(aers, 'r') as scores:
        aaer1 = scores.readline()
        aaer2 = scores.readline().strip().strip('[').strip(']').replace(',','').split()
        for i in xrange(len(aaer2)):
            aaer2[i] = float(aaer2[i])
else:
    aaer1 = []
    aaer2 = []
    n = 0
    fname = os.path.join(directory, "dev.atts.out" + str(n))
    while n != maxm and os.path.isfile(fname):
        attention_pickle = fname
        aer1, aer2 = aer6.run(target_algs, text_in, text_out, attention_pickle)
        aaer1.append(aer1)
        aaer2.append(aer2)
        n += 1
        fname = os.path.join(directory, "dev.atts.out" + str(n))
        if not os.path.isfile(fname):
            n+=1
            fname = os.path.join(directory, "dev.atts.out" + str(n))


    out_file = os.path.join(directory, 'aerscores2')
    open(out_file, 'w').close()
    with open(out_file, 'w+') as out:
        out.write(str(aaer1))
        out.write("\n")
        out.write(str(aaer2))

aers = os.path.join(baseline, "aerscores2")
if os.path.isfile(aers):
    with open(aers, 'r') as scores:
        aaer1b = scores.readline()
        aaer2b = scores.readline().strip().strip('[').strip(']').replace(',','').split()
        for i in xrange(len(aaer2b)):
            aaer2b[i] = float(aaer2b[i])
#plot
plt.plot(step, aaer2, 'r-', stepb, aaer2b, 'g-')
plt.axis([0, 100000, 0.45, 0.8])
plt.title("AER score progression with training steps (method 2)")
plt.xlabel("step")
plt.ylabel("AER")
plt.savefig(os.path.join(directory, 'aer2_bas.png'))
#plt.plot(step, aaer2, 'r-')
#plt.savefig(os.path.join(directory, 'aer26.png'))

plt.clf()



# Calculate & plot AER

aers = os.path.join(directory, "aerscores")
if os.path.isfile(aers):
    with open(aers, 'r') as scores:
        aaer1 = scores.readline()
        aaer2 = scores.readline().strip().strip('[').strip(']').replace(',','').split()
        for i in xrange(len(aaer2)):
            aaer2[i] = float(aaer2[i])
else:
    aaer1 = []
    aaer2 = []

    n = 0
    fname = os.path.join(directory, "dev.atts.out" + str(n))
    while n != maxm and os.path.isfile(fname):
        attention_pickle = fname
        aer1, aer2 = aer5.run(target_algs, text_in, text_out, attention_pickle)
        aaer1.append(aer1)
        aaer2.append(aer2)
        n += 1
        fname = os.path.join(directory, "dev.atts.out" + str(n))
        if not os.path.isfile(fname):
            n+=1
            fname = os.path.join(directory, "dev.atts.out" + str(n))


    out_file = os.path.join(directory, 'aerscores')
    open(out_file, 'w').close()
    with open(out_file, 'w+') as out:
        out.write(str(aaer1))
        out.write("\n")
        out.write(str(aaer2))

aers = os.path.join(baseline, "aerscores")
if os.path.isfile(aers):
    with open(aers, 'r') as scores:
        aaer1b = scores.readline()
        aaer2b = scores.readline().strip().strip('[').strip(']').replace(',','').split()
        for i in xrange(len(aaer2b)):
            aaer2b[i] = float(aaer2b[i])
#plot
plt.plot(step, aaer2, 'b-', stepb, aaer2b, 'g-')
plt.axis([0, 100000, 0.45, 0.8])
plt.title("AER score progression with training steps (method 1)")
plt.xlabel("step")
plt.ylabel("AER")
plt.savefig(os.path.join(directory, 'aer1_bas.png'))
#plt.plot(step, aaer2, 'r-')
#plt.savefig(os.path.join(directory, 'aer26.png'))

