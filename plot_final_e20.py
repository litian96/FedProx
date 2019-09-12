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

        search_train_accu = re.search(r'At round (.*) training accuracy: (.*)', line, re.M | re.I)
        if search_train_accu:
            rounds.append(int(search_train_accu.group(1)))
        else:
            search_test_accu = re.search(r'At round (.*) accuracy: (.*)', line, re.M | re.I)
            if search_test_accu:
                accu.append(float(search_test_accu.group(2)))

        search_loss = re.search(r'At round (.*) training loss: (.*)', line, re.M | re.I)
        if search_loss:
            loss.append(float(search_loss.group(2)))

        search_loss = re.search(r'gradient difference: (.*)', line, re.M | re.I)
        if search_loss:
            sim.append(float(search_loss.group(1)))

    return rounds, sim, loss, accu


f = plt.figure(figsize=[23, 10])

log = ["synthetic_1_1", "mnist", "femnist", "shakespeare", "sent140_772user"]
titles = ["Synthetic", "MNIST", "FEMNIST", "Shakespeare", "Sent140"]
rounds = [200, 100, 200, 40, 800]
mus=[1, 1, 1, 0.001, 0.01]
drop_rates=[0, 0.5, 0.9]

sampling_rate = [1, 1, 2, 1, 10]
labels = ['FedAvg', r'FedProx ($\mu$=0)', r'FedProx ($\mu$>0)']

improv = 0

for drop_rate in range(3):
    for idx in range(5):

        ax = plt.subplot(3, 5, 5*(drop_rate)+idx+1)

        if drop_rate == 0:
            rounds1, sim1, losses1, test_accuracies1 = parse_log(log[idx] + "/fedprox_drop0_mu0")
        else:
            rounds1, sim1, losses1, test_accuracies1 = parse_log(log[idx] + "/fedavg_drop"+str(drop_rates[drop_rate]))
        rounds2, sim2, losses2, test_accuracies2 = parse_log(log[idx] + "/fedprox_drop"+str(drop_rates[drop_rate])+"_mu0")
        rounds3, sim3, losses3, test_accuracies3 = parse_log(log[idx] + "/fedprox_drop"+str(drop_rates[drop_rate])+"_mu" + str(mus[idx]))


        if sys.argv[1] == 'loss':
            if drop_rate == 2 and idx == 4:
                plt.plot(np.asarray(rounds1[:len(losses1):sampling_rate[idx]]), np.asarray(losses1)[::sampling_rate[idx]], ":", linewidth=3.0, label=labels[0], color="#ff7f0e")
                plt.plot(np.asarray(rounds2[:len(losses2):sampling_rate[idx]]), np.asarray(losses2)[::sampling_rate[idx]], '--', linewidth=3.0, label=labels[1], color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(losses3):sampling_rate[idx]]), np.asarray(losses3)[::sampling_rate[idx]], linewidth=3.0, label=labels[2], color="#17becf")
            else:
                plt.plot(np.asarray(rounds1[:len(losses1):sampling_rate[idx]]),
                         np.asarray(losses1)[::sampling_rate[idx]], ":", linewidth=3.0, color="#ff7f0e")
                plt.plot(np.asarray(rounds2[:len(losses2):sampling_rate[idx]]),
                         np.asarray(losses2)[::sampling_rate[idx]], '--', linewidth=3.0, color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(losses3):sampling_rate[idx]]),
                         np.asarray(losses3)[::sampling_rate[idx]], linewidth=3.0, color="#17becf")

        elif sys.argv[1] == 'accuracy':
            if drop_rate == 2 and idx == 4:
                plt.plot(np.asarray(rounds1[:len(test_accuracies1):sampling_rate[idx]]), np.asarray(test_accuracies1)[::sampling_rate[idx]], ":", linewidth=3.0, label=labels[0], color="#ff7f0e")
                plt.plot(np.asarray(rounds2[:len(test_accuracies2):sampling_rate[idx]]), np.asarray(test_accuracies2)[::sampling_rate[idx]], '--', linewidth=3.0, label=labels[1], color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(test_accuracies3):sampling_rate[idx]]), np.asarray(test_accuracies3)[::sampling_rate[idx]], linewidth=3.0, label=labels[2], color="#17becf")
            else:
                plt.plot(np.asarray(rounds1[:len(test_accuracies1):sampling_rate[idx]]),
                         np.asarray(test_accuracies1)[::sampling_rate[idx]], ":", linewidth=3.0, color="#ff7f0e")
                plt.plot(np.asarray(rounds2[:len(test_accuracies2):sampling_rate[idx]]),
                         np.asarray(test_accuracies2)[::sampling_rate[idx]], '--', linewidth=3.0, color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(test_accuracies3):sampling_rate[idx]]),
                         np.asarray(test_accuracies3)[::sampling_rate[idx]], linewidth=3.0, color="#17becf")

        plt.xlabel("# Rounds", fontsize=22)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)

        if sys.argv[1] == 'loss' and idx == 0:
            plt.ylabel('Training Loss', fontsize=22)
        elif sys.argv[1] == 'accuracy' and idx == 0:
            plt.ylabel('Testing Accuracy', fontsize=22)

        if drop_rate == 0:
            plt.title(titles[idx], fontsize=22, fontweight='bold')

        ax.tick_params(color='#dddddd')
        ax.spines['bottom'].set_color('#dddddd')
        ax.spines['top'].set_color('#dddddd')
        ax.spines['right'].set_color('#dddddd')
        ax.spines['left'].set_color('#dddddd')
        ax.set_xlim(0, rounds[idx])

        if sys.argv[1] == "loss":
            if drop_rate == 0 and idx == 4:
                plt.ylim(0.3, 1.8)
            if drop_rate == 0 and idx == 1:
                plt.ylim(0.2, 1.5)
            if drop_rate == 1 and idx == 1:
                plt.ylim(0.2, 1.5)
            if drop_rate == 2 and idx == 1:
                plt.ylim(0.2, 1.5)
            if drop_rate == 2 and idx == 4:
                plt.ylim(0.2, 4)
            if drop_rate == 1 and idx == 4:
                plt.ylim(0.2, 1.8)


f.legend(frameon=False, loc='lower center', ncol=3, prop=dict(weight='bold'), borderaxespad=-0.3, fontsize=26)  # note: different from plt.legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
f.savefig(sys.argv[1] + "_full.pdf")
