##### PACKAGES

from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import math # math.isnan
import os
import time
import psutil
from torch.utils.tensorboard import SummaryWriter # needed for displaying in tensorboard (need to do pip install tensorboard. needs to be above specific version.)
import argparse

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor # stable_baselines3 requires gym==0.17.3
from wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames
from functions import GMM_OPE

import msgpack
from msgpack_numpy import patch as mspack_numpy_patch
mspack_numpy_patch() # needed to save the trained model.

from A1_dqn_train import nature_cnn, DQN
from torch.distributions.normal import Normal
import pickle
import matplotlib.pyplot as plt
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters for OPE Experiment")
    parser.add_argument('--game', type=str, help='Name of the Atari game environment')
    parser.add_argument('--epsilon_behavior', type=float, help='Epsilon value used in behavior policy')
    parser.add_argument('--seedno', type=int, default=1, help='Random seed number')
    parser.add_argument('--sample_size', type=int, help='Sample size')
    parser.add_argument('--Num_Iterations', type=int, help='Number of training iterations; set to 0 or None for infinite loop')
    parser.add_argument('--method', type=str, choices=["Energy", "RBF", "PDFL2", "KL", "FLE", "Hyvarinen", "TVD"], help='Divergence or kernel method to use')
    # parser.add_argument('--sigma_laplace', type=float, default=1.0, help='Laplace: sigma of Laplace kernel')
    # parser.add_argument('--gridno_laplace', type=int, default=100, help='Lapalce: number of grids for grid-approximation')   
    parser.add_argument('--sigma_rbf', type=float, default=1.0, help='RBF: sigma of RBF kernel')
    parser.add_argument('--pdfL2_eps', type=float, default=1.0, help='PDFL2: prevents variance explosion')
    parser.add_argument('--KL_resample_mixture', type=int, default=100, help='KL: number of resamples from a single conditional distribution (previous iterate)') # Stage9 (stochastic policy)
    parser.add_argument('--KL_eps', type=float, default=1.0, help='KL: prevents variance shrinkage')
    parser.add_argument('--Hyvarinen_resample_mixture', type=int, default=100, help='Hyvarinen: number of resamples from a single conditional distribution (previous iterate)') # Stage9 (stochastic policy)
    parser.add_argument('--Hyvarinen_eps', type=float, default=10.0, help='Hyvarinen: prevents variance shrinkage')  # For stability, used bigger values than 1.0.
    parser.add_argument('--TVD_resample_mixture', type=int, default=100, help='TVD: number of resamples from a single conditional distribution (previous iterate)') # Stage9 (stochastic policy)
    parser.add_argument('--TVD_eps', type=float, default=1.0, help='TVD: prevents variance shrinkage')
    # parser.add_argument('--QRDQN_kappa', type=float, default=0.01, help='QRDQN: kappa in Huber loss')
    parser.add_argument('--num_mixture', type=int, help='Number of mixtures in GMM model')   
    parser.add_argument('--epsilon_target', type=float, help='Epsilon of target policy')
    parser.add_argument('--reward_var', type=float, help='Variance of reward')    
    return parser.parse_args()


if __name__ == "__main__": 

    args = parse_args()
    game = args.game
    epsilon_behavior = args.epsilon_behavior
    seedno = args.seedno
    Num_Iterations = args.Num_Iterations  # SEPARATE FROM SAMPLE_SIZE.
    method = args.method
    SAMPLE_SIZE=args.sample_size

    # sigma_laplace=args.sigma_laplace; gridno=args.gridno_laplace                 # Laplace: grid-approximation
    sigma_rbf=args.sigma_rbf                                                     # RBF
    pdfL2_eps = args.pdfL2_eps                                                   # PDFL2: prevents variance-sum (denominator) from shrinking to zero.
    KL_resample_mixture = args.KL_resample_mixture; KL_eps = args.KL_eps         # KL: sample from each mixture component, prevent density value from blowing up.

    Hyvarinen_resample_mixture=args.Hyvarinen_resample_mixture
    Hyvarinen_eps=args.Hyvarinen_eps
    TVD_resample_mixture = args.TVD_resample_mixture
    TVD_eps = args.TVD_eps    
    # QRDQN_kappa = args.QRDQN_kappa

    num_mixture = args.num_mixture

    reward_var=args.reward_var
    epsilon_target=args.epsilon_target

    # reward_var=1. ; epsilon_target=0.3      # Stage=8 (Breakout)
    # reward_var=1. ; epsilon_target=0.0      # Stage=8 (non-Breakout)
    

    ## Hard-coded settings

    GAMMA=0.95; reward_mean = 0.; reward_min=-10.0; reward_max=10.0; nonzero_reward_multiply=20 # Hard-coded tuning parameters.

    BATCH_SIZE=32; NUM_ENVS = 1; LR = 1e-2; min_sq_grad=None; TARGET_UPDATE_FREQ = 1000
    LOG_DIR = "dpi_values_rewardvar" + str(int(reward_var)) + "/logs/N" + str(SAMPLE_SIZE) + "/" + game + "_BehaviorEps" + str(int(epsilon_behavior*10)) + "/" + "Mix" + str(num_mixture) + "/" + method + "_seed" + str(seedno) + "/" ; LOG_INTERVAL = 1000; use_cuda = True; dummy_or_subproc = "subproc"


    # plot_histogram=True
    plot_histogram=False # Turn off for simulations. 

    cnn_freeze=True # Freeze the CNN part.
    # cnn_freeze=False

    # float_type = torch.float32
    float_type = torch.float64

    if use_cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    print('\n\n\n-------')
    print('game: ', game)
    print('Epsilon-behavior: {:.1f}'.format(epsilon_behavior))
    print('mixture: ', num_mixture)
    print('device:', device)
    print('seed=', seedno)
    print('method=', method)
    print('-------\n\n\n')


    ### Optimal policy and states from dpi

    input_envs = [lambda: make_atari_deepmind(game, seed=i, scale_values=True) for i in range(NUM_ENVS)]

    if dummy_or_subproc=="dummy":
        vec_env = DummyVecEnv(input_envs) 
    elif dummy_or_subproc=="subproc":
        vec_env = SubprocVecEnv(input_envs) 
    else:
        raise ValueError("dummy_or_subproc must be either 'dummy' or 'subproc'")    

    env = BatchedPytorchFrameStack(vec_env, k=4) 

    optimal_policy = DQN(env, device)
    optimal_policy = optimal_policy.to(device)
    optimal_policy.load('./optimal/model/' + game + '.pack')
    optimal_policy = optimal_policy.to(float_type) 


    # states_filename = "dpi_values/" + game + "_Gamma" + str(int(GAMMA*100)) + ".pkl"
    states_filename = "dpi_values_rewardvar" + str(int(reward_var)) + "/" + game + "_Gamma" + str(int(GAMMA*100)) + ".pkl"

    with open(states_filename, "rb") as f:
        dict_dpi = pickle.load(f)

    obses_dpi = dict_dpi['states']
    obses_dpi = np.stack([o.get_frames() for o in obses_dpi])  
    obses_dpi = torch.from_numpy(obses_dpi).to(device=device, dtype=float_type)
    M_inaccuracy = obses_dpi.shape[0]

    actions_dpi = dict_dpi['actions']
    actions_dpi = torch.as_tensor(actions_dpi, dtype=torch.int64, device=device).unsqueeze(-1)
    actions_dpi = actions_dpi.unsqueeze(-1).unsqueeze(-1)              # (M_inaccuracy, 1, 1, 1)
    actions_dpi = actions_dpi.expand(-1, 1, num_mixture, 3)            # (M_inaccuracy, 1, num_mix, 3)

    # print(obses_dpi); print(actions_dpi); print(returns_dpi)
    # print(actions_dpi == optimal_policy.act(obses_dpi, epsilon=epsilon_target)) # verity that actions are chosen correctly.

    returns_dpi = dict_dpi['returns']
    M_inaccuracy = len(returns_dpi)
    returns_dpi = torch.tensor(returns_dpi, dtype=float_type)
    returns_dpi, _ = torch.sort(returns_dpi) # sort them before we measure Wasserstein



    ### Setting the network

    if seedno is not None: # https://hoya012.github.io/blog/reproducible_pytorch/ contains exact explanation.
        # seedno = 42184871 * seedno + 342034023
        torch.manual_seed(seedno)
        # env.seed(seed) # common seed. but we are going to use different seeds for each env.
        env.action_space.seed(seedno) 
        random.seed(seedno)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    replay_buffer = deque(maxlen=SAMPLE_SIZE)


    ### Collect samples.

    start_time = time.time()

    prev_life=0 # can be arbitrary
    new_stage=True

    obses = env.reset()
    for sample_ind in range(SAMPLE_SIZE):

        act_obses = np.stack([o.get_frames() for o in obses]) # 4 (NUM_ENVS), 4 (stacked), 84, 84
        actions=optimal_policy.act(act_obses, epsilon=epsilon_behavior, dtype=float_type)
        # actions = online_net.behavior_act(act_obses, epsilon_behavior)

        if new_stage:
            actions=[1] # For Breakout.

        new_obses, rews, dones, infos = env.step(actions)

        life = infos[0]['ale.lives']
        if life < prev_life:
            new_stage=True
            # new_obses=env.reset()
        else:
            new_stage=False
        prev_life = life


        # if rews[0]!=0:
        rews[0] = max(reward_min, min(rews[0], reward_max)) * nonzero_reward_multiply  # clip  & multiply
        rews[0] += random.gauss(reward_mean, math.sqrt(reward_var))       # add noise

        for obs, action, rew, _, new_obs, _ in zip(obses, actions, rews, dones, new_obses, infos):
            transition = (obs, action, rew, new_obs)
            replay_buffer.append(transition)

        obses = new_obses

        if (sample_ind + 1) % 1000==0:
            print("{} / {} samples generated.".format(len(replay_buffer), SAMPLE_SIZE))
    print("Sample generation completed.\n\n")


    ### Main Training Loop

    # obses = env.reset() 
    # exit()

    if plot_histogram:
        plt.ion()  
        fig, ax = plt.subplots()

    while True:

        try:

            summary_writer = SummaryWriter(LOG_DIR)  # makes a log file. 

            online_net = GMM_OPE(env=env, optimal_policy=optimal_policy, epsilon_target=epsilon_target, device=device, method=method, num_mixture=num_mixture, float_type=float_type, GAMMA=GAMMA, pdfL2_eps=pdfL2_eps, sigma_rbf=sigma_rbf, KL_eps=KL_eps, KL_resample_mixture=KL_resample_mixture, Hyvarinen_resample_mixture=Hyvarinen_resample_mixture, Hyvarinen_eps=Hyvarinen_eps, TVD_resample_mixture=TVD_resample_mixture, TVD_eps=TVD_eps)
            target_net = GMM_OPE(env=env, optimal_policy=optimal_policy, epsilon_target=epsilon_target, device=device, method=method, num_mixture=num_mixture, float_type=float_type, GAMMA=GAMMA, pdfL2_eps=pdfL2_eps, sigma_rbf=sigma_rbf, KL_eps=KL_eps, KL_resample_mixture=KL_resample_mixture, Hyvarinen_resample_mixture=Hyvarinen_resample_mixture, Hyvarinen_eps=Hyvarinen_eps, TVD_resample_mixture=TVD_resample_mixture, TVD_eps=TVD_eps) 

            online_net.net[0].load_state_dict(optimal_policy.net[0].state_dict()) # Borrow the first part of optimal policy (conv_net)
            target_net.net[0].load_state_dict(optimal_policy.net[0].state_dict())

            online_net = online_net.to(device).to(float_type) 
            target_net = target_net.to(device).to(float_type)


            if cnn_freeze:
                for param in online_net.net[0].parameters():
                    param.requires_grad = False

                for param in target_net.net[0].parameters():
                    param.requires_grad = False


            if min_sq_grad is None:
                optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)
            else:
                optimizer = torch.optim.Adam(online_net.parameters(), lr=LR, eps=min_sq_grad)


            for step in range(Num_Iterations+1):

                ## Start Gradient Descent
                transitions = random.sample(replay_buffer, BATCH_SIZE) 
                if method == "Energy" or method == "RBF":
                    loss = online_net.MMD_loss(transitions, target_net) 
                elif method == "PDFL2":
                    loss = online_net.PDFL2_loss(transitions, target_net) 
                elif method == "KL":
                    loss = online_net.KL(transitions, target_net) 
                elif method == "FLE":
                    loss = online_net.FLE(transitions, target_net) 
                elif method == "Hyvarinen":
                    loss = online_net.Hyvarinen(transitions, target_net) 
                elif method == "TVD":
                    loss = online_net.TVD(transitions, target_net) 
                # elif method == "QRDQN":
                #     loss = online_net.QRDQN(transitions, target_net) 

                # print("step="+str(step)); print(loss)
                # exit()


                ## Gradient Descent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                ## Update Target Net
                if step % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(online_net.state_dict())


                ### Compute Wasserstein-1 inaccuracy (approximated).

                if step % LOG_INTERVAL == 0:

                    with torch.no_grad():

                        # if method=="QRDQN":

                        #     online_out = online_net(obses_dpi)
                        #     online_out =  online_out.view(M_inaccuracy, env.action_space.n, num_mixture)
                        #     actions_dpi = dict_dpi['actions']                            
                        #     actions_dpi = torch.tensor(actions_dpi, device=online_out.device)
                        #     online_selected = online_out[torch.arange(online_out.size(0)), actions_dpi]
                        #     # print(online_out.shape)
                        #     # print(len(actions_dpi))
                        #     # print(online_selected.shape)

                        #     chosen_idx = torch.randint(0, num_mixture, (M_inaccuracy,))
                        #     return_samples_inaccuracy = online_selected[torch.arange(M_inaccuracy), chosen_idx]
                        #     # print(return_samples_inaccuracy.shape)
                        #     return_samples_inaccuracy=return_samples_inaccuracy.tolist()


                        # else:
                        online_out = online_net(obses_dpi)
                        online_out =  online_out.view(M_inaccuracy, env.action_space.n, num_mixture, 3)
                        online_selected = online_out.gather(1, actions_dpi).squeeze(1)     # (M_inaccuracy, num_mix, 3)

                        return_samples_inaccuracy = []
                        for i in range(M_inaccuracy):
                            ## Extract 10 mixture components for the i-th sample
                            mixtures = online_selected[i]  # shape: [10, 3]

                            ## Get normalized weights
                            log_weights = mixtures[:, 0]
                            # log_weights = log_weights - log_weights.max() # Prevent blowing up ton inf when exponentiated.
                            # weights = torch.exp(log_weights)
                            # weights /= weights.sum()
                            # print(weights.sum())

                            if not torch.all(torch.isfinite(log_weights)):  # If restart the model. 
                                shutil.rmtree(LOG_DIR)
                                print("\nseedno"+str(seedno)+" Step="+str(step) +" has NaN values : Restart iterations.\n")
                                raise ValueError("seedno"+str(seedno)+" Step="+str(step) +" has NaN values : Restart iterations.")




                            weights = torch.softmax(log_weights, dim=-1)

                            ## Sample a component index based on weights
                            chosen_idx = torch.multinomial(weights, 1).item()
                            chosen_component = mixtures[chosen_idx]
                            # print(chosen_idx); print(chosen_component)

                            ## Extract mean and std from the selected component
                            mean = chosen_component[1].item()
                            log_var = chosen_component[2].item()
                            std = math.sqrt(math.exp(log_var))

                            ## Sample from N(mean, std**2)
                            sample = random.gauss(mean, std)
                            return_samples_inaccuracy.append(sample)

                            # print(len(return_samples_inaccuracy)==M_inaccuracy)

                        return_samples_inaccuracy = torch.tensor(return_samples_inaccuracy)
                        return_samples_inaccuracy, _ = torch.sort(return_samples_inaccuracy) # sort them before we measure Wasserstein
                        wasserstein_inaccuracy = torch.mean(torch.abs(return_samples_inaccuracy - returns_dpi)).item()
                        print("Step="+str(step))
                        print("(Marginalized) Wasserstein-dpi inaccuracy : {:.10f}".format(wasserstein_inaccuracy))


                        histogram_predict = return_samples_inaccuracy.tolist()
                        has_inf = any(math.isinf(x) for x in histogram_predict)
                        if has_inf:
                            print("Warning: predictions have infinity values.")
                        histogram_predict = [x for x in histogram_predict if math.isfinite(x)]


                        if plot_histogram:

                            ax.clear()  
                            ax.hist(returns_dpi, edgecolor='blue', alpha=0.5, density=True, label='True', color='blue')
                            ax.hist(histogram_predict, edgecolor='orange', alpha=0.5, density=True, label='Predicted', color='orange')

                            ax.set_title(
                                f"Step {step} - Predicted vs True (Marginal) Return Distribution\n"
                                f"(Marginalized) Wasserstein-dpi inaccuracy : {wasserstein_inaccuracy:.4f}"
                            )

                            ax.set_xlabel("Sampled Return")
                            ax.set_ylabel("Relative Frequency")
                            ax.grid(True)
                            ax.legend()  
                            fig.canvas.draw()  # If we update the plots within for-loop, we must add these.
                            fig.canvas.flush_events()

                        end_time=time.time()
                        print("Cumulative Time (secs) : {:.4f}".format(end_time - start_time))

                        memory_usage=psutil.Process(os.getpid()).memory_info().rss/1024**3
                        print('Current Memory (GB): {:.4f}'.format(memory_usage))
                        print("\n")

                    summary_writer.add_scalar('Wasserstein_Inaccuracy', wasserstein_inaccuracy, global_step=step)

            print("\nSeed=" + str(seedno) + " : Final (Marginalized) inaccuracy : {:.10f}\n".format(wasserstein_inaccuracy))

            break

        except ValueError as e:
            continue


# tensorboard --logdir ./dpi_values/logs/BreakoutNoFrameskip-v4_BehaviorEps1
# tensorboard --logdir ./previous/dpi_values_stage3/logs/
# tensorboard --logdir ./dpi_values/logs/


