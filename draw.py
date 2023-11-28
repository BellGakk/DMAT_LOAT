import matplotlib.pyplot as plt
import numpy as np
import argparse
import csv
import os

X = ['plane', 'car', 'bird', 'cat', 'dear', 'dog', 'frog', 'horse', 'ship', 'truck']
Y = [0.607, 0.747, 0.355, 0.252, 0.332, 0.445, 0.626, 0.638, 0.743, 0.677]
Y_reg_mse = [0.553, 0.728, 0.283, 0.104, 0.176, 0.547, 0.718, 0.606, 0.780, 0.633]

plt.figure(figsize=(25,25))
plt.title('LinfPGD-40', size=60)
plt.plot(X, Y_reg_mse, linewidth=2.0, markersize=20, label='MART+MORE-SE')
plt.plot(X, Y, linewidth=2.0, markersize=20, label='MART')

plt.xlabel('Classes', fontsize=60)
plt.ylabel('Adversarial Accuracy', fontsize=60)
   
plt.grid(color='gray', linestyle='dashed', linewidth=1.4, alpha=1.0)
plt.tick_params(axis='both', which='major', labelsize=45)

plt.ylim(0, 1)
plt.legend(loc='best', fontsize=60)
plt.savefig('resnet18_mart_fairness_pgd40.png', dpi=300)