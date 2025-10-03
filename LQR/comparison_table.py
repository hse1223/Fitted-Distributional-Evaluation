import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for OPE Experiment")
    parser.add_argument('--setting', type=int, help='setting')    
    parser.add_argument('--cheat_initial', action='store_true', help="whether to allow setting the true next value as the intial value in each iteration.")
    return parser.parse_args()

args = parse_args()

setting = args.setting
cheat_initial = args.cheat_initial


import os
import pickle
import re
import numpy as np
import pandas as pd

methods = ['Energy', 'Laplace', 'RBF', 'PDFL2', 'KL', 'FLE'] 
# methods = ['Energy', 'Laplace', 'RBF', 'PDFL2', 'KL']

if cheat_initial:
    setting= str(setting) + "_cheatedinitial"

results_dir = "Results/setting"+str(setting)
starting_point = 0 # Only include Nsize >= this value

# Initialize nested dicts
mean_table = {}
std_table = {}

for method in methods:
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

                    if nsize not in mean_table:
                        mean_table[nsize] = {}
                        std_table[nsize] = {}
                    mean_table[nsize][method] = mean_val
                    std_table[nsize][method] = std_val

    # if method=="Laplace":
    #     print(inaccuracies)
    #     print(len(inaccuracies))
    #     print(mean_val)
    #     print(std_val)
    #     exit()

# Convert to DataFrames
df_mean = pd.DataFrame.from_dict(mean_table, orient='index').sort_index()
df_std = pd.DataFrame.from_dict(std_table, orient='index').sort_index()

# Round for clarity
df_mean = df_mean.round(4)
df_std = df_std.round(4)

# Combine for final display table
rows = []
for nsize in df_mean.index:
    mean_row = df_mean.loc[nsize]
    std_row = df_std.loc[nsize].apply(lambda x: f"({x})")
    rows.append(mean_row)
    rows.append(std_row)

# Create final display DataFrame
combined_df = pd.DataFrame(rows)
combined_df.index = [idx for nsize in df_mean.index for idx in (nsize, nsize)]
combined_df.index.name = "Nsize"

# Display result
print(combined_df)
