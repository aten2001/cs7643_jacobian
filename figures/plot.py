# This script plots energy per atom vs. timesteps for each temperature.

import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import matplotlib.pyplot as plt
# from cycler import cycler
import math

n_proj_neg_1 = ["2019-11-25-16-53-00-jr-proj-n1.log", "2019-11-25-19-33-58-jr-proj-n1.log", "2019-11-25-22-15-09-jr-proj-n1.log", 
           "2019-11-26-00-55-24-jr-proj-n1.log", "2019-11-26-03-36-42-jr-proj-n1.log"]
n_proj_3 = ["2019-11-26-04-11-15-jr-proj-n3.log", "2019-11-26-06-03-46-jr-proj-n3.log", "2019-11-26-07-41-00-jr-proj-n3.log", 
           "2019-11-26-09-04-44-jr-proj-n3.log", "2019-11-26-10-28-26-jr-proj-n3.log"]
n_proj_1 = ["2019-11-26-12-46-03-jr-proj-n1.log", "2019-11-26-13-56-49-jr-proj-n1.log", "2019-11-26-15-08-06-jr-proj-n1.log", 
           "2019-11-26-16-07-21-jr-proj-n1.log", "2019-11-26-17-03-25-jr-proj-n1.log"]
for i in range(5):
    n_proj_neg_1[i] = "../log/" + n_proj_neg_1[i]
    n_proj_3[i] = "../log/" + n_proj_3[i]
    n_proj_1[i] = "../log/" + n_proj_1[i]

def get_accuracy(filenames): 
    accuracy = np.zeros((5, 240))
    J_norm = np.zeros((5, 240))
    for i in range(5):
        filename = filenames[i]
        if os.stat(filename).st_size==0:
            print "Error: log.lammps is empty at " + folder_directory
            continue

        file = open(filename, "r")
        line_count = 0
        for line in file:
            if line_count > 2400:
                break
            data = line.split()
            if line_count % 10 == 9:
                accuracy[i][int(line_count/10)] = float(data[7][:-1])
                J_norm[i][int(line_count/10)] = float(data[13][:-1])
                
            line_count += 1
        file.close()
        
    accuracy = accuracy * 100
    ave_acc = np.average(accuracy, axis=0)
    acc_error = np.std(accuracy, axis=0)
    ave_J_norm = np.average(J_norm, axis=0)
    J_norm_error = np.std(J_norm, axis=0)
    return ave_acc, acc_error, ave_J_norm, J_norm_error


ave_acc_1, acc_error_1, ave_J_norm_1, J_norm_error_1 = get_accuracy(n_proj_1)
ave_acc_3, acc_error_3, ave_J_norm_3, J_norm_error_3 = get_accuracy(n_proj_3)
ave_acc_neg_1, acc_error_neg_1, ave_J_norm_neg_1, J_norm_error_neg_1 = get_accuracy(n_proj_neg_1)


# plot Figure2a in /figures

plt.figure(figsize=(14,7))
plt.xlabel("Iteration")
plt.ylabel("Test Accuracy")
plt.ylim(98, 99.5)
plt.xlim(0, 15e4)

plt.plot(np.arange(0, 15e4, 15e4/240), ave_acc_1, linewidth=2, label = "n_proj=1", color='blue')
plt.fill_between(np.arange(0, 15e4, 15e4/240), ave_acc_1 - acc_error_1, ave_acc_1 + acc_error_1, color='cyan')

plt.plot(np.arange(0, 15e4, 15e4/240), ave_acc_3, linewidth=2, label = "n_proj=3", color='black')
plt.fill_between(np.arange(0, 15e4, 15e4/240), ave_acc_3 - acc_error_3, ave_acc_3 + acc_error_3, color='grey')

plt.plot(np.arange(0, 15e4, 15e4/240), ave_acc_neg_1, linewidth=2, label = "Exact", color='orangered')
plt.fill_between(np.arange(0, 15e4, 15e4/240), ave_acc_neg_1 - acc_error_neg_1, ave_acc_neg_1 + acc_error_neg_1, color='orange')

plt.legend(loc=4)
plt.savefig("Figure2a.png")


# plot Figure2b in /figures
plt.figure(figsize=(14,7))
plt.xlabel("Iteration")
plt.ylabel("Test Jacobian norm")
plt.ylim(0.5, 2)
plt.xlim(0, 15e4)

plt.plot(np.arange(0, 15e4, 15e4/240), ave_J_norm_1, linewidth=2, label = "n_proj=1", color='blue')
plt.fill_between(np.arange(0, 15e4, 15e4/240), ave_J_norm_1 - J_norm_error_1, ave_J_norm_1 + J_norm_error_1, color='cyan')

plt.plot(np.arange(0, 15e4, 15e4/240), ave_J_norm_3, linewidth=2, label = "n_proj=3", color='black')
plt.fill_between(np.arange(0, 15e4, 15e4/240), ave_J_norm_3 - J_norm_error_3, ave_J_norm_3 + J_norm_error_3, color='grey')

plt.plot(np.arange(0, 15e4, 15e4/240), ave_J_norm_neg_1, linewidth=2, label = "Exact", color='orangered')
plt.fill_between(np.arange(0, 15e4, 15e4/240), ave_J_norm_neg_1 - J_norm_error_neg_1, ave_J_norm_neg_1 + J_norm_error_neg_1, color='orange')

plt.legend(loc=1)
plt.savefig("Figure2b.png")







