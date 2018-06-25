import sys, os
import matplotlib.pyplot as plt

directory = sys.argv[1]
maxm = int(sys.argv[2])

bleu = []
step = []

n = 0
fname = os.path.join(directory, "eugh_short")
with open(fname, 'r') as f:
    for line in f:
        line = line.strip().split()
        if line[0] == 'global':
            if line[2] != '2000':
                step.append(int(line[2]))
                n += 1
        elif line[0][0] == 'n' or line[0][0] == 's':
            bleu.append(float(line[1][:-1]))
            n += 1
        if n == maxm:
            break


plt.plot(step, bleu, 'b-')
plt.show()
