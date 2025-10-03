import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for OPE Experiment")
    parser.add_argument('--setting', type=int, help='setting')    
    parser.add_argument('--cheat_initial', action='store_true', help="whether to allow setting the true next value as the intial value in each iteration.")
    return parser.parse_args()

args = parse_args()

setting = args.setting
cheat_initial = args.cheat_initial

# setting=1
# setting=2
# setting=3
# setting=4

# cheating=True
# cheating=False

starting_point = 0


import os
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt

# methods = ['Energy', 'Laplace', 'RBF', 'PDFL2', 'KL', 'FLE'] 
# linestyles = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1)), (0, (5, 1))]
# markers = ['o', 's', 'D', '^', '*', 'x']
# colors = ["blue", "orange", "green", "red", "pink", "brown"]

methods = ['FLE', 'KL', 'Energy', 'Laplace', 'RBF', 'PDFL2']
linestyles = [(0, (5, 1)), "solid", 'dotted', 'dashed', 'dashdot', 'dotted']
markers    = ['x', '*', 'o', 's', 'D', '^']
colors = ["brown", "pink", "blue", "orange", "green", "red"]


if cheat_initial:
    setting= str(setting) + "_cheatedinitial"

results_dir = "Results/setting"+str(setting)



# colors = plt.cm.tab10.colors  # A palette of 10 distinct colors


plt.figure(figsize=(10, 6))

for i, method in enumerate(methods):
    nsizes = []
    mean_inaccuracies = []
    std_inaccuracies = []

    for filename in os.listdir(results_dir):
        if filename.startswith(method) and filename.endswith(".pkl"):
            match = re.search(r'Nsize(\d+)', filename)
            if match:
                nsize = int(match.group(1))
                if nsize < starting_point:
                    continue
                with open(os.path.join(results_dir, filename), "rb") as f:
                    inaccuracies = pickle.load(f)
                    inaccuracies = [tup[0] for tup in inaccuracies] # final inaccuacy
                    # inaccuracies = [tup[1] for tup in inaccuracies] # later-half-mean inaccuracy
                    mean_val = np.mean(inaccuracies)
                    std_val = np.std(inaccuracies)
                    nsizes.append(nsize)
                    mean_inaccuracies.append(mean_val)
                    std_inaccuracies.append(std_val)

    # print(mean_inaccuracies)
    # print(std_inaccuracies)
    # print(len(inaccuracies))
    # exit()

    if not nsizes:
        continue

    # Sort data
    sorted_data = sorted(zip(nsizes, mean_inaccuracies, std_inaccuracies))
    nsizes, mean_inaccuracies, std_inaccuracies = zip(*sorted_data)

    mean_inaccuracies = np.array(mean_inaccuracies)
    std_inaccuracies = np.array(std_inaccuracies)

    # print(mean_inaccuracies)
    # print(std_inaccuracies)
    # exit()

    # Plot line and shaded region
    plt.plot(nsizes,
             mean_inaccuracies,
             linestyle=linestyles[i % len(linestyles)],
             marker=markers[i % len(markers)],
             color=colors[i % len(colors)],
             linewidth=4,
             label=method)

    simulation_num=len(inaccuracies)

    plt.fill_between(nsizes,
                     mean_inaccuracies - 2 * std_inaccuracies / np.sqrt(simulation_num),
                     mean_inaccuracies + 2 * std_inaccuracies / np.sqrt(simulation_num),
                     color=colors[i % len(colors)],
                     alpha=0.15)


# Final touches
plt.xlabel("Sample Size", fontsize=20)
plt.ylabel("Inaccuracy (log-scale)", fontsize=20)
# plt.title(f"Inaccuracy : Setting"+str(args.setting) )
plt.grid(True)
plt.yscale("log")
plt.legend()
plt.tight_layout()

if not os.path.exists('Results'):
    os.makedirs('Results')

# plt.show()
plt.savefig('Results/setting'+str(setting)+".png")

