import matplotlib
matplotlib.use('Agg')
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


def run(directory, n = 100, lamb = 1):
    print(lamb)
    no_encountered = 0
    no_plotted = 0
    total_loss = []
    likel_loss = []
    added = []
    log_file = os.path.join(directory, "ough")
    with open(log_file, 'r') as f:
        for line in f:
            if line[0] == "L" and line[1] == " ":
                com = line.find(",")
                loss1 = line[4:com]
                loss2 = line[com+2:line.find(")")]
                if loss1!="nan" and loss2!="nan":
                    #print(loss1)
                    #print(loss2)
                    loss1 = float(loss1)
                    loss2 = float(loss2)
                    if no_encountered % n == 0:
                        total_loss.append(loss1)
                        likel_loss.append(loss1-loss2)
                        added.append(loss2)
                        no_plotted += 1
                    else:
                        total_loss[no_plotted-1]+=loss1
                        likel_loss[no_plotted-1]+=loss1-loss2
                        added[no_plotted-1]+=loss2
                    no_encountered += 1

    print(no_encountered)
    #step = range(no_encountered)
    step = range(0, len(added)*n, n)
    likel_loss = np.asarray(likel_loss)
    added = np.asarray(added)
    added = added/lamb

    line1, = plt.plot(step, likel_loss/n, 'b-', label="likelihood")
    line2, = plt.plot(step, added/n, 'r-', label="additional")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    # Create a legend for the first line.
    #first_legend = plt.legend(handles=[line1], loc=1)

    # Add the legend manually to the current Axes.
    #ax = plt.gca().add_artist(first_legend)

    # Create another legend for the second line.
    #plt.legend(handles=[line2], loc=4)



    #plt.plot(likel_loss/n, 'b-', added/n, 'r-')
    #plt.show()
    plt.savefig(os.path.join(directory, 'loss.png'))



if len(sys.argv) == 1:
    print("How about the file?")
elif len(sys.argv) == 2:
    f = sys.argv[1]
    run(f)
else:
    f = sys.argv[1]
    lamb = float(sys.argv[2])
    n = 100
    run(f,n, lamb)


