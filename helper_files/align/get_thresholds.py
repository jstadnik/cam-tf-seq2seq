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
atts_file = sys.argv[1]
maxm = int(sys.argv[2])

#sum_up = os.path.join(directory, 'sum_up')
#open(sum_up, 'w').close()



# add an axis and bleu
#n = 0
#step = []
#bleu = []
#fname2 = os.path.join(directory, "eugh_short")
#if os.path.isfile(fname2):
#    with open(fname2, 'r') as f:
#        for line in f:
#            line = line.strip().split()
#            if line[0] == 'global':
#                if int(line[2]) != 2000:
#                    step.append(int(line[2]))
#                    n += 1
#            elif line[0][0] == 'n' or line[0][0] == 's':
#                bleu.append(float(line[1][:-1]))
#                n += 1
#            if n == maxm:
#                break
#else:
#    start = int(sys.argv[3])
#    step = range(start, start+n*2000, 2000)

#plt.plot(step, bleu, 'b-')
#plt.title("BLEU score curve")
#plt.xlabel("step")
#plt.ylabel("BLEU")
#plt.savefig(os.path.join(directory, 'bleu.png'))
#plt.clf()

#with open(sum_up, 'w+') as sumup:
#    sumup.write("\n\n\nDirectory: ")
#    sumup.write(directory)
#    sumup.write("\nTotal steps trained: ")
#    sumup.write(str(step[-1]))
#    sumup.write("\nHighest BLEU achieved: ")
#    sumup.write(str(max(bleu)))
#    sumup.write("\nHighest BLEU achieved at step: ")
#    sumup.write(str(step[bleu.index(max(bleu))]))


# Calculate & plot AER6

#Get possible tresholds
value = []
i = 0
while i<=0.5:
    value.append(i)
    i+=0.05

#aers = os.path.join(directory, "aerscores2")
#if os.path.isfile(aers):
#    with open(aers, 'r') as scores:
#        aaer1 = scores.readline()
#        aaer2 = scores.readline().strip().strip('[').strip(']').replace(',','').split()
#        for i in xrange(len(aaer2)):
#            aaer2[i] = float(aaer2[i])
if True:
    aaer1 = []
    aaer2 = []
    n = 0
    for val in value:
        aer1, aer2 = aer6.run(target_algs, text_in, text_out, atts_file, value2=val)
        aaer1.append(aer1)
        aaer2.append(aer2)
        n += 1


print(aaer2)
#plot
plt.plot(value, aaer2, 'r-')
plt.title("AER score vs threshold")
plt.xlabel("threshold")
plt.ylabel("AER")
plt.savefig('aer2.png')
#plt.plot(step, aaer2, 'r-')
#plt.savefig(os.path.join(directory, 'aer26.png'))

plt.clf()

if True:
    aaer1 = []
    aaer2 = []
    n = 0
    for val in value:
        aer1, aer2 = aer5.run(target_algs, text_in, text_out, atts_file, value1=val)
        aaer1.append(aer1)
        aaer2.append(aer2)
        n += 1

print(aaer2)
plt.plot(value, aaer2, 'r-')
plt.title("AER score vs threshold")
plt.xlabel("threshold")
plt.ylabel("AER")
plt.savefig('aer1.png')
"""
with open(sum_up, 'a+') as sumup:
    sumup.write("\nBest AER26 achieved: ")
    sumup.write(str(min(aaer2)))
    sumup.write("\nBest AER26 achieved at step: ")
    sumup.write(str(step[aaer2.index(min(aaer2))]))
    sumup.write("\nBLEU when best AER26: ")
    sumup.write(str(bleu[aaer2.index(min(aaer2))]))
    sumup.write("\nAER26 when highest BLEU: ")
    sumup.write(str(aaer2[bleu.index(max(bleu))]))

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


    out_file = os.path.join(directory, 'aerscores')
    open(out_file, 'w').close()
    with open(out_file, 'w+') as out:
        out.write(str(aaer1))
        out.write("\n")
        out.write(str(aaer2))

#plot
plt.plot(step, aaer2, 'b-')
plt.title("AER score progression with training steps (method 1)")
plt.xlabel("step")
plt.ylabel("AER")
plt.savefig(os.path.join(directory, 'aer2.png'))
#plt.plot(step, aaer2, 'r-')
#plt.savefig(os.path.join(directory, 'aer26.png'))

with open(sum_up, 'a+') as sumup:
    sumup.write("\nBest AER2 achieved: ")
    sumup.write(str(min(aaer2)))
    sumup.write("\nBest AER2 achieved at step: ")
    sumup.write(str(step[aaer2.index(min(aaer2))]))
    sumup.write("\nBLEU when best AER2: ")
    sumup.write(str(bleu[aaer2.index(min(aaer2))]))
    sumup.write("\nAER2 when highest BLEU: ")
    sumup.write(str(aaer2[bleu.index(max(bleu))]))"""
