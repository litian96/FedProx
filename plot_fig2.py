import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axisartist.axislines import Subplot

matplotlib.rc('xtick', labelsize=17) 
matplotlib.rc('ytick', labelsize=17) 


def parse_log(file_name):
    rounds = []
    accu = []
    loss = []
    sim = []

    for line in open(file_name, 'r'):

        search_train_accu = re.search(r'At round (.*) training accuracy: (.*)', line, re.M|re.I)
        if search_train_accu:
            rounds.append(int(search_train_accu.group(1)))
        else:
            search_test_accu = re.search( r'At round (.*) accuracy: (.*)', line, re.M|re.I)
            if search_test_accu:
                accu.append(float(search_test_accu.group(2)))
            
        search_loss = re.search(r'At round (.*) training loss: (.*)', line, re.M|re.I)
        if search_loss:
            loss.append(float(search_loss.group(2)))

        search_loss = re.search(r'gradient difference: (.*)', line, re.M|re.I)
        if search_loss:
            sim.append(float(search_loss.group(1)))

    return rounds, sim, loss, accu


idx = 0
f = plt.figure(figsize=[20, 4])

for log in ["synthetic_iid", "synthetic_0_0", "synthetic_0.5_0.5", "synthetic_1_1"]:
    ax = plt.subplot(1, 4, idx+1)
    idx += 1
    rounds1, sim1, losses1, test_accuracies1 = parse_log("log_synthetic/" + log + "_client10_epoch20_mu0")
    rounds2, sim2, losses2, test_accuracies2 = parse_log("log_synthetic/" + log + "_client10_epoch20_mu1")
    
    if sys.argv[1] == 'loss':
        plt.plot(np.asarray(rounds1[:len(losses1)]), np.asarray(losses1), '--', linewidth=3.0, label='mu=0, E=20', color="#17becf")
        plt.plot(np.asarray(rounds2[:len(losses2)]), np.asarray(losses2), linewidth=3.0, label='mu=1, E=20', color="#e377c2")
    
    elif sys.argv[1] == 'accuracy':
        plt.plot(np.asarray(rounds1[:len(test_accuracies1)]), np.asarray(test_accuracies1), '--', linewidth=3.0, label='mu=0, E=20', color="#17becf")
        plt.plot(np.asarray(rounds2[:len(test_accuracies2)]), np.asarray(test_accuracies2), linewidth=3.0, label='mu=1, E=20', color="#e377c2")
    else:
        plt.plot(np.asarray(rounds1[:len(sim1)]), np.asarray(sim1), '--', linewidth=3.0, label='mu=0, E=20', color="#17becf")
        plt.plot(np.asarray(rounds2[:len(sim2)]), np.asarray(sim2), linewidth=3.0, label='mu=1, E=20', color="#e377c2")
    
    plt.xlabel("# Rounds", fontsize=22)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)

    if idx == 1:
        if sys.argv[1] == 'loss':
            plt.ylabel('Training Loss', fontsize=22)
        elif sys.argv[1] == 'accuracy':
            plt.ylabel('Testing Accuracy', fontsize=22)
        else:
            plt.ylabel("Variance of Local Grad.", fontsize=22)
    plt.title(log, fontsize=22)
    ax.tick_params(color='#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    ax.spines['top'].set_color('#dddddd') 
    ax.spines['right'].set_color('#dddddd')
    ax.spines['left'].set_color('#dddddd')

    
    if idx == 4:
        plt.legend(fontsize=22)
    plt.tight_layout()

f.savefig(sys.argv[1]+".pdf")
