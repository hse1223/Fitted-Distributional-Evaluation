import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict
import argparse
import math
from scipy import stats as ss
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for OPE Experiment")
    parser.add_argument('--directory', type=str, help='Directory name (dpi_values_rewardvar ... ))')  
    parser.add_argument('--rank', action='store_true', help="display rank (along with their means) or the original inaccuracy values.")  
    return parser.parse_args()

args = parse_args()
directory =  args.directory
rank = args.rank

# rank=True
# rank=False



# directory="dpi_values_rewardvar0"


### Recommend to maintain the following.

behavioreps_list=["small", "big"]
game_list = ['AtlantisNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'EnduroNoFrameskip-v4', 'KungFuMasterNoFrameskip-v4', 'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']
mixture_list = ["Mix10", "Mix100", "Mix200"]
logs_root = directory + "/logs/"
lst=os.listdir(logs_root)
Nsize_list=sorted([int(x[1:]) for x in lst])

tag = "Wasserstein_Inaccuracy"
method_list = ["FLE", "QRDQN", "IQN", "KL", "Energy", 'PDFL2', 'RBF', 'TVD', 'Hyvarinen']
# method_list = ["FLE", "KL", "Energy", 'PDFL2', 'RBF', 'TVD', 'Hyvarinen']



for behavioreps in behavioreps_list:

    inaccuracy_bigtable = []

    for game in game_list:
        # print("\n\n")
        # print(method_list)

        print("Behavior_eps="+behavioreps + ", Game="+game)

        inaccuracy_table = []

        for mixture in mixture_list:

            # print("Behavior_eps="+behavioreps + ", Game="+game + ", Mix=" + mixture[3:])

            for Nsize in Nsize_list:

                # print("Behavior_eps="+behavioreps + ", Game="+game + ", Mix=" + mixture[3:] + ", N=" + str(Nsize)  )

                possible_behavioreps = sorted([x for x in os.listdir(logs_root + "N" + str(Nsize) + "/") if x.startswith(game)])

                if behavioreps=="small":
                    result_directory = logs_root + "N" + str(Nsize) + "/" + possible_behavioreps[0] +"/"+str(mixture)
                elif behavioreps=="big":
                    result_directory = logs_root + "N" + str(Nsize) + "/" + possible_behavioreps[1] +"/"+str(mixture)


                method_dirs = [d for d in os.listdir(result_directory) if os.path.isdir(os.path.join(result_directory, d))]
                # method_current_list = [ method for method in method_list if any(d.startswith(method) for d in method_dirs)]
                method_current_list = method_list
                # print(method_current_list)

                inaccuracy_methods = []
                for method in method_current_list:
                    # method = method_current_list[0] # 이후에 for loop으로. 
                    files = [result_directory + "/" + x for x in method_dirs if x.startswith(method)]

                    if len(files)==0:
                        inaccuracy_methods.append("NA")
                        continue

                    inaccuracy_seeds = []

                    # # 1안
                    # for file in files:
                    #     # file=files[0]

                    #     if len(os.listdir(file))>1: # If we mistakenly ran two simulations for the same case, it will return an error.
                    #         raise AssertionError("Multiple files in " + file) 

                    #     event_acc = EventAccumulator(file)    
                    #     event_acc.Reload()
                    #     scalar_events = event_acc.Scalars(tag)
                    #     inacc=scalar_events[-1].value   
                    #     inaccuracy_seeds.append(inacc)

                    # # 2안 (there may be multiple event files)
                    for file in files:

                        event_files = sorted(os.listdir(file)) 
                        event_file = os.path.join(file, event_files[-1])
                        event_acc = EventAccumulator(event_file)

                        event_acc.Reload()
                        scalar_events = event_acc.Scalars(tag)
                        inacc=scalar_events[-1].value   
                        inaccuracy_seeds.append(inacc)


                    inaccuracy_value = sum(inaccuracy_seeds) / len(inaccuracy_seeds)
                    inaccuracy_value = round(inaccuracy_value, 2)

                    # if math.isinf(inaccuracy_value):
                    #     inaccuracy_value = ">1e+36"

                    if inaccuracy_value > 100:
                        inaccuracy_value = ">100"

                    inaccuracy_methods.append(inaccuracy_value)

                if rank:

                    converted = [
                        np.nan if x == 'NA' else 1e9 if x == '>100' else float(x)
                        for x in inaccuracy_methods
                    ]

                    masked = np.array(converted)
                    valid_mask = ~np.isnan(masked)  # Boolean mask for valid entries

                    ranks = np.empty_like(masked)
                    # ranks[valid_mask] = ss.rankdata(masked[valid_mask], method='min')
                    ranks[valid_mask] = ss.rankdata(masked[valid_mask], method='max') # max rank for tie values.
                    ranks[~valid_mask] = np.nan  # Restore np.nan for originally NA values

                    inaccuracy_methods = ranks

                if not rank:
                    print(inaccuracy_methods)
                else:
                    inaccuracy_table.append(inaccuracy_methods)
                    inaccuracy_bigtable.append(inaccuracy_methods)

        if rank:
            inaccuracy_table = pd.DataFrame(inaccuracy_table)
            inaccuracy_table.columns = method_list
            print(round(inaccuracy_table.mean().to_frame().T, 3))


    if rank:
        print('\n>>> Big rank-comparison for ' + directory + " and Behavior-eps="+behavioreps+" : ")
        inaccuracy_bigtable = pd.DataFrame(inaccuracy_bigtable)
        inaccuracy_bigtable.columns = method_list
        print(round(inaccuracy_bigtable.mean().to_frame().T, 3))
        print("\n\n")

