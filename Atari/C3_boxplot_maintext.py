import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict
import argparse
import math
from scipy import stats as ss
from matplotlib.patches import Patch

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for OPE Experiment")
    parser.add_argument('--directory', type=str, help='Directory name (dpi_values_rewardvar ... ))')    
    parser.add_argument('--behavioreps', type=str, choices=["small", "big"], help='Behavior epsilon')    
    parser.add_argument('--Nsize', type=int, help='sample size')    
    parser.add_argument('--Mix', type=int, help='mixture number')    
    return parser.parse_args()

args = parse_args()
directory =  args.directory
Nsize = args.Nsize
behavioreps = args.behavioreps
mix_num = args.Mix

# rank=True
# rank=False


### Recommend to maintain the following.

# behavioreps_list=["small", "big"]
# game_list = ['AtlantisNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'EnduroNoFrameskip-v4', 'KungFuMasterNoFrameskip-v4', 'PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']

two_game_list = [['AtlantisNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'EnduroNoFrameskip-v4', 'KungFuMasterNoFrameskip-v4'], ['PongNoFrameskip-v4', 'QbertNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4']]

# mixture_list = ["Mix10", "Mix100", "Mix200"]
mixture_list = ["Mix" + str(mix_num)]
logs_root = directory + "/logs/"

# lst=os.listdir(logs_root)
# Nsize_list=sorted([int(x[1:]) for x in lst])

tag = "Wasserstein_Inaccuracy"

if mix_num < 200:
    method_list = ["FLE", "QRDQN", "IQN", "KL", "Energy", 'PDFL2', 'RBF', 'TVD', 'Hyvarinen']
else:
    method_list = ["FLE", "QRDQN", "IQN", "KL", "Energy", 'PDFL2', 'RBF', 'TVD']

num_mixes = len(mixture_list)
num_methods = len(method_list)

# directory_path = "plots/" + directory.removeprefix("dpi_values_") + "/" + "Eps_" + behavioreps + "/N" + str(int(Nsize)) + "/"
# os.makedirs(directory_path, exist_ok=True)


box_width = 0.2
spacing = 1  # Space between method groups
colors = ['skyblue', 'lightgreen', 'salmon']


# fig, axes = plt.subplots(nrows=1, ncols=len(game_list), figsize=(3 * len(game_list), 4), sharey=False)


image_list=[]
for plot_idx, game_list in enumerate(two_game_list):

    num_games = len(game_list)
    ncols = num_games
    # nrows = math.ceil(num_games / ncols)  # will be 2 if num_games = 7
    nrows=1

    # print(ncols, nrows)
    # print(game_list)
    # exit()


    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), sharey=False)
    axes = axes.flatten()  # make it 1D for easier indexing

    for i, game in enumerate(game_list):

        # print(i, game)

        ax = axes[i]

        inaccuracy_mixtures = []
        for mixture in mixture_list:

            possible_behavioreps = sorted([x for x in os.listdir(logs_root + "N" + str(Nsize) + "/") if x.startswith(game)])

            if behavioreps=="small":
                result_directory = logs_root + "N" + str(Nsize) + "/" + possible_behavioreps[0] +"/"+str(mixture)
            elif behavioreps=="big":
                result_directory = logs_root + "N" + str(Nsize) + "/" + possible_behavioreps[1] +"/"+str(mixture)


            method_dirs = [d for d in os.listdir(result_directory) if os.path.isdir(os.path.join(result_directory, d))]
            method_current_list = method_list

            inaccuracy_methods = []
            for method in method_current_list:
                files = [result_directory + "/" + x for x in method_dirs if x.startswith(method)]

                if len(files)==0:
                    inaccuracy_methods.append("NA")
                    continue

                inaccuracy_seeds = []

                for file in files:

                    event_files = sorted(os.listdir(file)) 
                    event_file = os.path.join(file, event_files[-1])
                    event_acc = EventAccumulator(event_file)

                    event_acc.Reload()
                    scalar_events = event_acc.Scalars(tag)
                    inacc=scalar_events[-1].value   

                    if math.isinf(inacc):
                        for event in reversed(scalar_events):
                            if math.isfinite(event.value):
                                inacc = event.value
                                break

                    inaccuracy_seeds.append(inacc)

                inaccuracy_methods.append(inaccuracy_seeds)

            inaccuracy_mixtures.append(inaccuracy_methods)


        # Step 1: Collect non-outlier min/max values (whiskers)

        flattened_values = []

        for mix in inaccuracy_mixtures:
            for method_data in mix:
                if method_data == 'NA':
                    continue
                flattened_values.extend(method_data)

        # Step 2: Set y-limits based on whiskers
        sorted_flattened_values = sorted(flattened_values, reverse=True)
        y_max = sorted_flattened_values[3]
        y_min = min(sorted_flattened_values)


        # Step 3: Prepare figure
        # fig, ax = plt.subplots()
        # fig, ax = plt.subplots(figsize=(15, 6))
        # fig, ax = plt.subplots(figsize=(15, 3)) # fine.
        # fig, ax = plt.subplots(figsize=(15, 2.5))

        # Step 4: Draw only mean ± std (no boxplots)
        for method_idx in range(num_methods):
            for mix_idx in range(num_mixes):
                value = inaccuracy_mixtures[mix_idx][method_idx]
                if value == 'NA':
                    continue

                group_x = method_idx * spacing
                x_pos = group_x + (mix_idx - 1) * box_width

                # Compute mean and std
                mean = np.mean(value)
                std = np.std(value)

                # Plot mean ± std interval as a thin vertical line
                ax.vlines(x_pos, mean - std, mean + std, color=colors[mix_idx], linewidth=1)

                # Plot mean as a solid dot
                ax.plot(x_pos, mean, 'o', color=colors[mix_idx], markersize=6)

        # Step 5: Format x-axis
        xticks = [i * spacing for i in range(num_methods)]
        ax.set_xticks(xticks)
        # ax.set_xticklabels(method_list, rotation=90, ha='right')
        ax.set_xticklabels(method_list, rotation=90, ha='right', fontsize=14)

        # Step 7: Final plot settings

        # ax.set_title(game)
        # ax.set_ylabel("Inaccuracy")
        ax.set_title(game, fontsize=15)      
        ax.set_ylabel("Inaccuracy", fontsize=14) 

        ax.tick_params(axis='y', labelsize=12)

        ax.set_ylim(bottom=y_min, top=y_max * 1.05)
        ax.grid(True, axis='y')


    # fig.delaxes(axes[-1])

    plt.tight_layout()

    # plt.suptitle("Mixture="+str(mix_num), fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.95]) 

    # plt.suptitle("Mixture="+str(mix_num), fontsize=16, y=0.93)
    # plt.tight_layout(rect=[0, 0, 1, 0.90])  # leave space for the title

    # plt.show()

    # Save
    os.makedirs("plot_latex", exist_ok=True)
    image_name="plot_latex/" + "Mix" + str(mix_num) + "_rearragned_index" + str(plot_idx+1) + ".png"
    plt.savefig(image_name)
    image_list.append(image_name)




from PIL import Image

# Load the images
img1 = Image.open(image_list[0])
img2 = Image.open(image_list[1])

# Ensure both have the same width for clean vertical stacking
max_width = max(img1.width, img2.width)
total_height = img1.height + img2.height

# Create a blank canvas (with transparent or white background)
new_img = Image.new("RGBA", (max_width, total_height), (255, 255, 255, 0))  # (white background + transparent alpha)

# Compute horizontal offsets to center them
offset1 = ((max_width - img1.width) // 2, 0)
offset2 = ((max_width - img2.width) // 2, img1.height)

# Paste images
new_img.paste(img1, offset1)
new_img.paste(img2, offset2)

# Save to file
new_img.save("plot_latex/Mix"+str(mix_num)+"_combined.png")


