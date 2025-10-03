# Fitted-Distributional-Evaluation
This repository contains the Python codes for the paper &lt;A Principled Path to Fitted Distributional Evaluation> that is accepted by NeurlIPS 2025.
- LQR (Linear Quadratic Regulator): We compare FLE (baseline) and our FDE (fitted distributional evaluation) methods (KL, Energy, Laplace, RBF, PDFL2).
- Atari games: We compare FLE, QRDQN, IQN, TVD (baseline) and our FDE methods (KL, Energy, PDFL2, RBF, Hyvarinen). We compare these in seven different game settings (Atlantis, Breakout, Enduro, Kungfumaster, Pong, Qbert, Spaceinvader) under four different settings (weak / strong coverage, deterministic / random reward).

Following image represents the progress of estimation throughout iterations. Using one of our FDE methods called Energy FDE, our estimation (yellow histogram) converges towards the true distribution (blue histogram) in an Atari game called Breakout.

<img width="1990" height="421" alt="Breakout_combined" src="https://github.com/user-attachments/assets/3c2a670a-aa1a-41b1-aa5c-0049aaae6fed" />


# How to run the codes
## LQR
- Run cmd.sh to obtain the same plots in the paper.
## Atari
- **cmdA_optimal.sh** applies deep-Q learning to train an optimal policy. However, we recommend not to run this, since the trained policies are already saved in the directory optimal/.
- **cmdB_GMM.sh** applies GMM (Gaussian mixture model) based methods. These include baseline methods (FLE, TVD) and our FDE methods (KL, PDFL2, Energy, RBF).
- **cmdB_Particle.sh** applies particle (dirac delta) based methods. These include the quantile based methods (QRDQN, IQN).
- **cmdD_table.sh** obtains the inaccuracy values and their ranks for each setting.
- **cmdE_boxplot.sh** dispalys the inaccuracy comparison via boxplots.
## Python version
- We are using Python==3.9.13 with the following modules.

gym==0.17.3

numpy==1.21.5

stable_baselines3==1.6.2

torch==1.13.1+cu116

psutil==5.9.0

msgpack==1.0.3

matplotlib==3.6.2

tensorboard==2.11.0

scipy==1.9.3

pandas==1.4.4


