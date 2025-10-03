# Fitted-Distributional-Evaluation
This repository contains the Python codes for the paper &lt;A Principled Path to Fitted Distributional Evaluation> that is accepted by NeurlIPS 2025.
- LQR (Linear Quadratic Regulator): We compare FLE (baseline) and our FDE (fitted distributional evaluation) methods (KL, Energy, Laplace, RBF, PDFL2).
- Atari games: We compare FLE, QRDQN, IQN, TVD (baseline) and our FDE methods (KL, Energy, PDFL2, RBF, Hyvarinen). We compare these in seven different game settings (Atlantis, Breakout, Enduro, Kungfumaster, Pong, Qbert, Spaceinvader) under four different settings (weak / strong coverage, deterministic / random reward).

Following image represents the progress of estimation throughout iterations. Using one of our FDE methods called Energy FDE, our estimation (yellow histogram) converges towards the true distribution (blue histogram) in an Atari game called Breakout.

<img width="1990" height="421" alt="Breakout_combined" src="https://github.com/user-attachments/assets/3c2a670a-aa1a-41b1-aa5c-0049aaae6fed" />




