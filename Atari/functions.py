from torch import nn
import torch
# import gym
# from functools import partial
from A1_dqn_train import nature_cnn, DQN
from torch.distributions.normal import Normal
import numpy as np
import math



class GMM_OPE(nn.Module):
    def __init__(self, env, optimal_policy, epsilon_target, device, method, num_mixture, float_type, GAMMA, pdfL2_eps, sigma_rbf, KL_eps, KL_resample_mixture, Hyvarinen_resample_mixture, Hyvarinen_eps, TVD_resample_mixture, TVD_eps):
        super().__init__()

        self.num_actions = env.action_space.n
        self.optimal_policy = optimal_policy
        self.device = device 

        self.num_mixture = num_mixture
        self.float_type = float_type
        self.GAMMA = GAMMA
        self.pdfL2_eps = pdfL2_eps
        self.epsilon_target = epsilon_target

        # self.sigma_laplace = sigma_laplace
        # self.gridno = gridno
        self.sigma_rbf = sigma_rbf
        self.KL_eps = KL_eps
        self.KL_resample_mixture = KL_resample_mixture

        self.TVD_resample_mixture = TVD_resample_mixture
        self.TVD_eps = TVD_eps  

        self.Hyvarinen_resample_mixture = Hyvarinen_resample_mixture
        self.Hyvarinen_eps = Hyvarinen_eps

        if method=="Energy":
            self.K0=self.K0_energy
        elif method=="Laplace":
            self.K0=self.K0_laplace
        elif method=="RBF":
            self.K0=self.K0_rbf


        conv_net = nature_cnn(env.observation_space)

        # ## First trial (shallow)
        # self.fc = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.num_actions * self.num_mixture * 3)   # output: mb_size x action_num x num_mix x 3 (log_weight, mu, log_var)
        # )

        ## Second trial (deepened)
        self.fc = nn.Sequential(
            nn.Linear(512, 450),
            nn.ReLU(),
            nn.Linear(450, 400),
            nn.ReLU(),
            nn.Linear(400, 350),
            nn.ReLU(),
            nn.Linear(350, 300),
            nn.ReLU(),
            nn.Linear(300, 250),
            nn.ReLU(),
            nn.Linear(250, 200),
            nn.ReLU(),
            nn.Linear(200, 150),
            nn.ReLU(),
            nn.Linear(150, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions * self.num_mixture * 3)   # output: mb_size x action_num x num_mix x 3 (log_weight, mu, log_var)
        )

        # ## Third trial (include quantile-approach)
        # self.huberloss=torch.nn.HuberLoss(reduction='none', delta=QRDQN_kappa)
        # self.fc = nn.Sequential(
        #     nn.Linear(512, 450),
        #     nn.ReLU(),
        #     nn.Linear(450, 400),
        #     nn.ReLU(),
        #     nn.Linear(400, 350),
        #     nn.ReLU(),
        #     nn.Linear(350, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 250),
        #     nn.ReLU(),
        #     nn.Linear(250, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 150),
        #     nn.ReLU(),
        #     nn.Linear(150, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.num_actions * self.num_mixture)   
        # )

        self.net = nn.Sequential(conv_net, self.fc)

    def forward(self, x):
        return self.net(x)

    def behavior_act(self, obses, epsilon_behavior): # behavior policy

        actions=self.optimal_policy.act(obses, epsilon=epsilon_behavior, dtype=self.float_type)
        return actions

    def MMD_loss(self, transitions, target_net):

        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        new_obses = [t[3] for t in transitions]

        obses = np.stack([o.get_frames() for o in obses])
        new_obses = np.stack([o.get_frames() for o in new_obses])

        obses_t = torch.as_tensor(obses, dtype=self.float_type, device=self.device)                         # (mb_size, 4, 84, 84)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)         # (mb_size, 1)
        rews_t = torch.as_tensor(rews, dtype=self.float_type, device=self.device).unsqueeze(-1)             # (mb_size, 1)
        new_obses_t = torch.as_tensor(new_obses, dtype=self.float_type, device=self.device)                 # (mb_size, 4, 84, 84)

        new_actions = self.optimal_policy.act(new_obses_t, epsilon=self.epsilon_target, dtype=self.float_type)
        new_actions_t = torch.as_tensor(new_actions, dtype=torch.int64, device=self.device).unsqueeze(-1) # (mb_size, 1)
        mb_size = obses_t.shape[0]

        # ---- Step 1: Forward pass ----
        output = self.net(obses_t)  # (mb_size, num_actions * num_mix * 3)
        output = output.view(mb_size, self.num_actions, self.num_mixture, 3) # (mb_size, num_actions, num_mix, 3)

        # ---- Step 3: Select by action ----
        action_indices = actions_t.unsqueeze(-1).unsqueeze(-1)  # shape: (mb_size, 1, 1, 1)
        action_indices = action_indices.expand(-1, 1, self.num_mixture, 3)  # (mb_size, 1, num_mix, 3)
        selected = output.gather(1, action_indices).squeeze(1)  # (mb_size, num_mix, 3)
        # print(selected.shape)
 
        # ---- Step 4: Extract and process parameters ----
        log_weights = selected[..., 0]  # (mb_size, num_mix)
        # log_weights = log_weights - log_weights.max() # Prevent exploding to inf value when exponentiated.
        means = selected[..., 1]        # (mb_size, num_mix)
        log_vars = selected[..., 2]     # (mb_size, num_mix)

        # weights = torch.exp(log_weights)
        vars_ = torch.exp(log_vars)
        # weights = weights / torch.sum(weights, dim=1, keepdim=True)
        weights = torch.softmax(log_weights, dim=-1)


        # ---- Step 5: Get target net values (for s', a') ----

        with torch.no_grad():
            target_out = target_net(new_obses_t)
            target_out = target_out.view(mb_size, self.num_actions, self.num_mixture, 3)

            new_action_indices = new_actions_t.unsqueeze(-1).unsqueeze(-1)  # shape: (mb_size, 1, 1, 1)
            new_action_indices = new_action_indices.expand(-1, 1, self.num_mixture, 3)  # (mb_size, 1, num_mix-target, 3)
            target_selected = target_out.gather(1, new_action_indices).squeeze(1)  # (mb_size, num_mix-target, 3)
            # print(target_selected.shape)

            target_log_weights = target_selected[..., 0]
            # target_log_weights = target_log_weights - target_log_weights.max() # Prevent blowing up to infinity when exponentiated.
            target_means = target_selected[..., 1]
            target_log_vars = target_selected[..., 2]

            # target_weights = torch.exp(target_log_weights)
            target_vars = torch.exp(target_log_vars)
            # target_weights = target_weights / torch.sum(target_weights, dim=1, keepdim=True)
            target_weights = torch.softmax(target_log_weights, dim=-1)


        # ---- Step 6: Compute the MMD loss ----

        ## Term 1 (solely based on online network)
        mu1 = means.unsqueeze(2)         # (mb_size, num_mix-online1, 1)
        mu2 = means.unsqueeze(1)         # (mb_size, 1, num_mix-online2)
        var1 = vars_.unsqueeze(2)
        var2 = vars_.unsqueeze(1)

        mu_diff = mu1 - mu2              # (mb_size, num_mix-online1, num_mix-online2)
        sum_var = var1 + var2            # (mb_size, num_mix-online1, num_mix-online2)
        # print(mu_diff.shape); print(sum_var.shape)

        K0_self = self.K0(mu_diff, sum_var)                      # (mb_size, num_mix-online1, num_mix-online2)
        w_prod = weights.unsqueeze(2) * weights.unsqueeze(1)     # (mb_size, num_mix-online1, num_mix-online2)
        Term1 = (w_prod * K0_self).sum(dim=(1, 2))               # (mb_size,)


        ## Term 2 (online & target network)
        mu1_cross = means.unsqueeze(2)                           # (mb_size, num_mix-online, 1)
        mu2_cross = (rews_t + self.GAMMA * target_means).unsqueeze(1) # (mb_size, 1, num_mix-target)

        var1_cross = vars_.unsqueeze(2)                          # (mb_size, num_mix-online, 1)
        var2_cross = (self.GAMMA ** 2) * target_vars.unsqueeze(1)     # (mb_size, 1, num_mix-target)

        mu_diff_cross = mu1_cross - mu2_cross                    # (mb_size, num_mix-online, num_mix-target)
        sum_var_cross = var1_cross + var2_cross                  # (mb_size, num_mix-online, num_mix-target)

        K0_cross = self.K0(mu_diff_cross, sum_var_cross)         # (mb_size, num_mix-online, num_mix-target)
        w_prod_cross = weights.unsqueeze(2) * target_weights.unsqueeze(1) # (mb_size, num_mix-online, num_mix-target)
        Term2 = (w_prod_cross * K0_cross).sum(dim=(1, 2))        # (mb_size,)

        ## MMD
        mmd = Term1 - 2 * Term2
        loss = mmd.mean()

        return loss


    def PDFL2_loss(self, transitions, target_net):

        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        new_obses = [t[3] for t in transitions]

        obses = np.stack([o.get_frames() for o in obses])
        new_obses = np.stack([o.get_frames() for o in new_obses])

        obses_t = torch.as_tensor(obses, dtype=self.float_type, device=self.device)                         # (mb_size, 4, 84, 84)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)         # (mb_size, 1)
        rews_t = torch.as_tensor(rews, dtype=self.float_type, device=self.device).unsqueeze(-1)             # (mb_size, 1)
        new_obses_t = torch.as_tensor(new_obses, dtype=self.float_type, device=self.device)                 # (mb_size, 4, 84, 84)

        new_actions = self.optimal_policy.act(new_obses_t, epsilon=self.epsilon_target, dtype=self.float_type)
        new_actions_t = torch.as_tensor(new_actions, dtype=torch.int64, device=self.device).unsqueeze(-1) # (mb_size, 1)
        mb_size = obses_t.shape[0]

        # ---- Step 1: Forward pass ----
        output = self.net(obses_t)  # (mb_size, num_actions * num_mix * 3)
        output = output.view(mb_size, self.num_actions, self.num_mixture, 3) # (mb_size, num_actions, num_mix, 3)

        # ---- Step 3: Select by action ----
        action_indices = actions_t.unsqueeze(-1).unsqueeze(-1)  # shape: (mb_size, 1, 1, 1)
        action_indices = action_indices.expand(-1, 1, self.num_mixture, 3)  # (mb_size, 1, num_mix, 3)
        selected = output.gather(1, action_indices).squeeze(1)  # (mb_size, num_mix, 3)
        # print(selected.shape)
 
        # ---- Step 4: Extract and process parameters ----
        log_weights = selected[..., 0]  # (mb_size, num_mix)
        # log_weights = log_weights - log_weights.max() # Prevent exploding to inf value when exponentiated.
        means = selected[..., 1]        # (mb_size, num_mix)
        log_vars = selected[..., 2]     # (mb_size, num_mix)

        # weights = torch.exp(log_weights)
        vars_ = torch.exp(log_vars)
        # print(weights.shape); print(vars_.shape)
        # weights = weights / torch.sum(weights, dim=1, keepdim=True)
        # print(weights.sum(dim=1))
        weights = torch.softmax(log_weights, dim=-1)


        # ---- Step 5: Get target net values (for s', a') ----
        with torch.no_grad():
            target_out = target_net(new_obses_t)
            target_out = target_out.view(mb_size, self.num_actions, self.num_mixture, 3)

            new_action_indices = new_actions_t.unsqueeze(-1).unsqueeze(-1)  # shape: (mb_size, 1, 1, 1)
            new_action_indices = new_action_indices.expand(-1, 1, self.num_mixture, 3)  # (mb_size, 1, num_mix-target, 3)
            target_selected = target_out.gather(1, new_action_indices).squeeze(1)  # (mb_size, num_mix-target, 3)
            # print(target_selected.shape)

            target_log_weights = target_selected[..., 0]
            # target_log_weights = target_log_weights - target_log_weights.max() # Prevent blowing up to infinity when exponentiated.
            target_means = target_selected[..., 1]
            target_log_vars = target_selected[..., 2]

            # target_weights = torch.exp(target_log_weights)
            target_vars = torch.exp(target_log_vars)
            # target_weights = target_weights / torch.sum(target_weights, dim=1, keepdim=True)
            target_weights = torch.softmax(target_log_weights, dim=-1)


        # ---- Step 6: Compute the MMD loss ----

        ## Term 1 (solely based on online network)
        mu1 = means.unsqueeze(2)         # (mb_size, num_mix-online1, 1)
        mu2 = means.unsqueeze(1)         # (mb_size, 1, num_mix-online2)
        var1 = vars_.unsqueeze(2)        # (mb_size, num_mix-online1, 1)
        var2 = vars_.unsqueeze(1)        # (mb_size, 1, num_mix-online2)
        w_prod = weights.unsqueeze(2) * weights.unsqueeze(1)     # (mb_size, num_mix-online1, num_mix-online2)
        var_sum = var1 + var2 + self.pdfL2_eps
        mu_diff = mu1 - mu2

        second_term_self = 1 / torch.sqrt(2 * math.pi * var_sum)
        third_term_self = torch.exp(- mu_diff**2 / (2*var_sum))
        tensor_indiv_self = w_prod * second_term_self * third_term_self  # (mb_size, num_mix-online1, num_mix-online2)
        Term1 = tensor_indiv_self.sum(dim=(1, 2)) 


        ## Term 2 (online & target network)
        mu1_cross = means.unsqueeze(2)                           # (mb_size, num_mix-online, 1)
        mu2_cross = (rews_t + self.GAMMA * target_means).unsqueeze(1) # (mb_size, 1, num_mix-target)
        var1_cross = vars_.unsqueeze(2)                          # (mb_size, num_mix-online, 1)
        var2_cross = (self.GAMMA ** 2) * target_vars.unsqueeze(1)     # (mb_size, 1, num_mix-target)
        w_prod_cross = weights.unsqueeze(2) * target_weights.unsqueeze(1) # (mb_size, num_mix-online, num_mix-target)

        var_sum_cross = var1_cross + var2_cross + self.pdfL2_eps
        mu_diff_cross = mu1_cross - mu2_cross
        second_term_cross = 1 / torch.sqrt(2 * math.pi * var_sum_cross)
        third_term_cross = torch.exp(- mu_diff_cross**2 / (2*var_sum_cross))
        tensor_indiv_cross = w_prod_cross * second_term_cross * third_term_cross
        Term2 = tensor_indiv_cross.sum(dim=(1, 2)) 

        pdfl2 = Term1 - 2 * Term2
        loss = pdfl2.mean()

        return loss


    def KL(self, transitions, target_net):   

        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        new_obses = [t[3] for t in transitions]

        obses = np.stack([o.get_frames() for o in obses])
        new_obses = np.stack([o.get_frames() for o in new_obses])

        obses_t = torch.as_tensor(obses, dtype=self.float_type, device=self.device)                         # (mb_size, 4, 84, 84)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)           # (mb_size, 1)
        rews_t = torch.as_tensor(rews, dtype=self.float_type, device=self.device).unsqueeze(-1)             # (mb_size, 1)
        new_obses_t = torch.as_tensor(new_obses, dtype=self.float_type, device=self.device)                 # (mb_size, 4, 84, 84)

        new_actions = self.optimal_policy.act(new_obses_t, epsilon=self.epsilon_target, dtype=self.float_type)
        new_actions_t = torch.as_tensor(new_actions, dtype=torch.int64, device=self.device).unsqueeze(-1) # (mb_size, 1)
        mb_size = obses_t.shape[0]

        # ---- Step 1: Forward pass ----
        output = self.net(obses_t)  # (mb_size, num_actions * num_mix * 3)
        output = output.view(mb_size, self.num_actions, self.num_mixture, 3) # (mb_size, num_actions, num_mix, 3)

        # ---- Step 3: Select by action ----
        action_indices = actions_t.unsqueeze(-1).unsqueeze(-1)  # shape: (mb_size, 1, 1, 1)
        action_indices = action_indices.expand(-1, 1, self.num_mixture, 3)  # (mb_size, 1, num_mix, 3)
        selected = output.gather(1, action_indices).squeeze(1)  # (mb_size, num_mix, 3)
        # print(selected.shape)
 
        # ---- Step 4: Extract and process parameters ----
        log_weights = selected[..., 0]  # (mb_size, num_mix)
        # log_weights = log_weights - log_weights.max() # Prevent exploding to inf value when exponentiated.
        means = selected[..., 1]        # (mb_size, num_mix)
        log_vars = selected[..., 2]     # (mb_size, num_mix)

        vars_ = torch.exp(log_vars)
        # weights = torch.exp(log_weights)
        # weights = weights / torch.sum(weights, dim=1, keepdim=True)
        weights = torch.softmax(log_weights, dim=-1)


        # ---- Step 5: Get target net values (for s', a') ----

        with torch.no_grad():
            target_out = target_net(new_obses_t)
            target_out = target_out.view(mb_size, self.num_actions, self.num_mixture, 3)

            new_action_indices = new_actions_t.unsqueeze(-1).unsqueeze(-1)  # shape: (mb_size, 1, 1, 1)
            new_action_indices = new_action_indices.expand(-1, 1, self.num_mixture, 3)  # (mb_size, 1, num_mix-target, 3)
            target_selected = target_out.gather(1, new_action_indices).squeeze(1)  # (mb_size, num_mix-target, 3)
            # print(target_selected.shape)

            target_log_weights = target_selected[..., 0]
            # target_log_weights = target_log_weights - target_log_weights.max() # Prevent blowing up to infinity when exponentiated.
            target_means = target_selected[..., 1]
            target_log_vars = target_selected[..., 2]

            target_vars = torch.exp(target_log_vars)
            # target_weights = torch.exp(target_log_weights)
            # target_weights = target_weights / torch.sum(target_weights, dim=1, keepdim=True)
            target_weights = torch.softmax(target_log_weights, dim=-1)


        # ---- Step 6: Compute the Likelihood (resample-approximated) ----

        # means   # (mb_size, num_mix-online)
        # vars_   # (mb_size, num_mix-online)
        # weights # (mb_size, num_mix-online)
        means_target = rews_t + self.GAMMA * target_means  # (mb_size, num_mix-target)
        vars_target = (self.GAMMA ** 2) * target_vars      # (mb_size, num_mix-target)
        # target_weights                                   # (mb_size, num_mix-target)

        ## Sample x from target policy
        means_target_exp = means_target.unsqueeze(2).expand(mb_size, self.num_mixture, self.KL_resample_mixture) # (mb_size, num_mix-target, B_samples)
        vars_target_exp = vars_target.unsqueeze(2).expand(mb_size, self.num_mixture, self.KL_resample_mixture) + self.KL_eps   # (mb_size, num_mix-target, B_samples)
        std_target = torch.sqrt(vars_target_exp)   # (mb_size, num_mix-target, B_samples)
        normal_samples = torch.randn_like(std_target)    # (mb_size, num_mix-target, B_samples)
        x_samples = means_target_exp + std_target * normal_samples  # (mb_size, num_mix_target, B)


        ## Compute φ(x | mu_online, sigma_online) for all samples
        x_samples_exp = x_samples.unsqueeze(-1)      # (mb_size, num_mix_target, B, 1)
        means_exp = means.unsqueeze(1).unsqueeze(1)  # (mb_size, 1, 1, num_mix_online)
        vars_exp = vars_.unsqueeze(1).unsqueeze(1) + self.KL_eps   # (mb_size, 1, 1, num_mix_online)
        log_norm_consts = -0.5 * torch.log(2 * math.pi * vars_exp)
        log_probs = log_norm_consts - 0.5 * ((x_samples_exp - means_exp)**2 / (vars_exp))  # (mb_size, num_mix_target, B, num_mix_online)

        ## Summations
        weights_exp = weights.unsqueeze(1).unsqueeze(1)          # (mb_size, 1, 1, num_mix_online)
        weighted_log_probs = log_probs + torch.log(weights_exp)  # (mb_size, num_mix_target, B, num_mix_online)
        log_gmm_online = torch.logsumexp(weighted_log_probs, dim=-1)  # (mb_size, num_mix_target, B)
        log_gmm_online_mean = log_gmm_online.mean(dim=-1)  # (mb_size, num_mix_target)
        weighted_sum = (target_weights * log_gmm_online_mean).sum(dim=-1)  # (mb_size,)

        loss = -weighted_sum.mean()
        return loss

    def FLE(self, transitions, target_net):  

        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        new_obses = [t[3] for t in transitions]

        obses = np.stack([o.get_frames() for o in obses])
        new_obses = np.stack([o.get_frames() for o in new_obses])

        obses_t = torch.as_tensor(obses, dtype=self.float_type, device=self.device)                         # (mb_size, 4, 84, 84)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)           # (mb_size, 1)
        rews_t = torch.as_tensor(rews, dtype=self.float_type, device=self.device).unsqueeze(-1)             # (mb_size, 1)
        new_obses_t = torch.as_tensor(new_obses, dtype=self.float_type, device=self.device)                 # (mb_size, 4, 84, 84)

        new_actions = self.optimal_policy.act(new_obses_t, epsilon=self.epsilon_target, dtype=self.float_type)
        new_actions_t = torch.as_tensor(new_actions, dtype=torch.int64, device=self.device).unsqueeze(-1) # (mb_size, 1)
        mb_size = obses_t.shape[0]

        # ---- Step 1: Forward pass ----
        output = self.net(obses_t)  # (mb_size, num_actions * num_mix * 3)
        output = output.view(mb_size, self.num_actions, self.num_mixture, 3) # (mb_size, num_actions, num_mix, 3)

        # ---- Step 3: Select by action ----
        action_indices = actions_t.unsqueeze(-1).unsqueeze(-1)  # shape: (mb_size, 1, 1, 1)
        action_indices = action_indices.expand(-1, 1, self.num_mixture, 3)  # (mb_size, 1, num_mix, 3)
        selected = output.gather(1, action_indices).squeeze(1)  # (mb_size, num_mix, 3)
        # print(selected.shape)
 
        # ---- Step 4: Extract and process parameters ----
        log_weights = selected[..., 0]  # (mb_size, num_mix)
        means = selected[..., 1]        # (mb_size, num_mix)
        log_vars = selected[..., 2]     # (mb_size, num_mix)

        vars_ = torch.exp(log_vars)
        weights = torch.softmax(log_weights, dim=-1)


        # ---- Step 5: Get target net values (for s', a') ----

        with torch.no_grad():
            target_out = target_net(new_obses_t)
            target_out = target_out.view(mb_size, self.num_actions, self.num_mixture, 3)

            new_action_indices = new_actions_t.unsqueeze(-1).unsqueeze(-1)  # shape: (mb_size, 1, 1, 1)
            new_action_indices = new_action_indices.expand(-1, 1, self.num_mixture, 3)  # (mb_size, 1, num_mix-target, 3)
            target_selected = target_out.gather(1, new_action_indices).squeeze(1)  # (mb_size, num_mix-target, 3)

            target_log_weights = target_selected[..., 0]
            target_means = target_selected[..., 1]
            target_log_vars = target_selected[..., 2]

            target_vars = torch.exp(target_log_vars)
            target_weights = torch.softmax(target_log_weights, dim=-1)


        # ---- Step 6: Compute the Likelihood (resample-approximated) ----

        # means   # (mb_size, num_mix-online)
        # vars_   # (mb_size, num_mix-online)
        # weights # (mb_size, num_mix-online)
        means_target = rews_t + self.GAMMA * target_means  # (mb_size, num_mix-target)
        vars_target = (self.GAMMA ** 2) * target_vars      # (mb_size, num_mix-target)
        # target_weights                                   # (mb_size, num_mix-target)


        ## Sample x from target network
        sampled_indices = torch.multinomial(target_weights, num_samples=1).squeeze(-1)  # (mb_size,)
        idx = sampled_indices.unsqueeze(-1)  # (mb_size, 1)
        means_j2 = torch.gather(means_target, 1, idx).squeeze(-1)  # (mb_size,)
        vars_j2 = torch.gather(vars_target, 1, idx).squeeze(-1) + self.KL_eps    # (mb_size,)        
        std_j2 = torch.sqrt(vars_j2)  # (mb_size,)
        noise = torch.randn_like(std_j2)       # (mb_size,)
        x_samples = means_j2 + std_j2 * noise          # (mb_size,)


        ## Compute φ(x | mu_online, sigma_online) for all samples
        x_samples = x_samples.unsqueeze(-1)                      # (mb_size, 1)
        means_exp = means                            # (mb_size, num_mix_online)
        vars_exp = vars_ + self.KL_eps                    # (mb_size, num_mix_online)        

        log_norm_consts = -0.5 * torch.log(2 * math.pi * vars_exp)
        log_probs = log_norm_consts - 0.5 * ((x_samples - means_exp)**2 / vars_exp)  # (mb_size, num_mix_online)


        log_weighted_probs = log_probs + torch.log(weights)  # (mb_size, num_mix_online)
        log_gmm_online = torch.logsumexp(log_weighted_probs, dim=-1)  # (mb_size,)
        loss = -log_gmm_online.mean()
        return loss


    def K0_energy(self, mean, var):
        std = torch.sqrt(var)
        normal = Normal(0, 1)
        abs_mean = torch.abs(mean)
        term1 = std * torch.sqrt(torch.tensor(2.0 / math.pi)) * torch.exp(-mean**2 / (2 * var))
        term2 = abs_mean * (1 - 2 * normal.cdf(-abs_mean / std))
        return -(term1 + term2)


    # def K0_laplace(mean, var, sigma_laplace, gridno): # grid-based approximation
    def K0_laplace(self, mean, var): # grid-based approximation

        sigma = torch.sqrt(var)
        device = mean.device

        z_grid = torch.linspace(-2.56, 2.56, steps=self.gridno, device=device)  # (gridno,)   => Great idea: no need to expand a grid-tensor.
        dz = z_grid[1] - z_grid[0]        # constant spacing in z
        z_grid = z_grid.view(1, 1, 1, -1) # (1,1,1,gridno)
        sigma = sigma.unsqueeze(-1)       # (mb_size, num_mix, num_mix, 1)
        mean = mean.unsqueeze(-1)         # (mb_size, num_mix, num_mix, 1)

        y = mean + sigma * z_grid  # (mb_size, num_mix, num_mix, gridno)
        k_vals = torch.exp(-torch.abs(y) / self.sigma_laplace)     # (mb_size, num_mix, num_mix, gridno)
        normal_pdf = torch.exp(-0.5 * z_grid**2) / math.sqrt(2 * math.pi)  # (1,1,1,gridno)

        integrand = k_vals * normal_pdf   # (mb_size, num_mix, num_mix, gridno)
        K0 = integrand.sum(dim=-1) * dz   # (mb_size, num_mix, num_mix)
        return K0


    def K0_rbf(self, mean, var):
        const = 1 / (torch.sqrt(torch.tensor(2 * math.pi)) * self.sigma_rbf)
        denom = torch.sqrt(1 + var / (2 * self.sigma_rbf**2))
        exp_term = torch.exp(-mean**2 / (4 * self.sigma_rbf**2 + 2 * var))
        return const / denom * exp_term
    

    def Hyvarinen(self, transitions, target_net):   

        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        new_obses = [t[3] for t in transitions]

        obses = np.stack([o.get_frames() for o in obses])
        new_obses = np.stack([o.get_frames() for o in new_obses])

        obses_t = torch.as_tensor(obses, dtype=self.float_type, device=self.device)                         # (mb_size, 4, 84, 84)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)           # (mb_size, 1)
        rews_t = torch.as_tensor(rews, dtype=self.float_type, device=self.device).unsqueeze(-1)             # (mb_size, 1)
        new_obses_t = torch.as_tensor(new_obses, dtype=self.float_type, device=self.device)                 # (mb_size, 4, 84, 84)

        new_actions = self.optimal_policy.act(new_obses_t, epsilon=self.epsilon_target, dtype=self.float_type)
        new_actions_t = torch.as_tensor(new_actions, dtype=torch.int64, device=self.device).unsqueeze(-1) # (mb_size, 1)
        mb_size = obses_t.shape[0]

        # ---- Step 1: Forward pass ----
        output = self.net(obses_t)  # (mb_size, num_actions * num_mix * 3)
        output = output.view(mb_size, self.num_actions, self.num_mixture, 3) # (mb_size, num_actions, num_mix, 3)

        # ---- Step 3: Select by action ----
        action_indices = actions_t.unsqueeze(-1).unsqueeze(-1)  # shape: (mb_size, 1, 1, 1)
        action_indices = action_indices.expand(-1, 1, self.num_mixture, 3)  # (mb_size, 1, num_mix, 3)
        selected = output.gather(1, action_indices).squeeze(1)  # (mb_size, num_mix, 3)
        # print(selected.shape)
 
        # ---- Step 4: Extract and process parameters ----
        log_weights = selected[..., 0]  # (mb_size, num_mix)
        # log_weights = log_weights - log_weights.max() # Prevent exploding to inf value when exponentiated.
        means = selected[..., 1]        # (mb_size, num_mix)
        log_vars = selected[..., 2]     # (mb_size, num_mix)

        vars_ = torch.exp(log_vars)
        # weights = torch.exp(log_weights)
        # weights = weights / torch.sum(weights, dim=1, keepdim=True)
        weights = torch.softmax(log_weights, dim=-1)


        # ---- Step 5: Get target net values (for s', a') ----

        with torch.no_grad():
            target_out = target_net(new_obses_t)
            target_out = target_out.view(mb_size, self.num_actions, self.num_mixture, 3)

            new_action_indices = new_actions_t.unsqueeze(-1).unsqueeze(-1)  # shape: (mb_size, 1, 1, 1)
            new_action_indices = new_action_indices.expand(-1, 1, self.num_mixture, 3)  # (mb_size, 1, num_mix-target, 3)
            target_selected = target_out.gather(1, new_action_indices).squeeze(1)  # (mb_size, num_mix-target, 3)
            # print(target_selected.shape)

            target_log_weights = target_selected[..., 0]
            # target_log_weights = target_log_weights - target_log_weights.max() # Prevent blowing up to infinity when exponentiated.
            target_means = target_selected[..., 1]
            target_log_vars = target_selected[..., 2]

            target_vars = torch.exp(target_log_vars)
            # target_weights = torch.exp(target_log_weights)
            # target_weights = target_weights / torch.sum(target_weights, dim=1, keepdim=True)
            target_weights = torch.softmax(target_log_weights, dim=-1)


        # ---- Step 6: Compute the Hyvarinen Loss (resample-approximated) ----

        # means   # (mb_size, num_mix-online)
        # vars_   # (mb_size, num_mix-online)
        # weights # (mb_size, num_mix-online)
        means_target = rews_t + self.GAMMA * target_means  # (mb_size, num_mix-target)
        vars_target = (self.GAMMA ** 2) * target_vars      # (mb_size, num_mix-target)
        # target_weights                                   # (mb_size, num_mix-target)

        ## Sample x from target policy
        means_target_exp = means_target.unsqueeze(2).expand(mb_size, self.num_mixture, self.Hyvarinen_resample_mixture) # (mb_size, num_mix-target, B_samples)
        vars_target_exp = vars_target.unsqueeze(2).expand(mb_size, self.num_mixture, self.Hyvarinen_resample_mixture) + self.Hyvarinen_eps   # (mb_size, num_mix-target, B_samples)
        std_target = torch.sqrt(vars_target_exp)   # (mb_size, num_mix-target, B_samples)
        normal_samples = torch.randn_like(std_target)    # (mb_size, num_mix-target, B_samples)
        x_samples = means_target_exp + std_target * normal_samples  # (mb_size, num_mix_target, B)


        # Expand for broadcasting
        x = x_samples.unsqueeze(2)                    # (mb_size, num_mix_target, 1, B)
        means = means.unsqueeze(1).unsqueeze(-1)      # (mb_size, 1, num_mix_online, 1)
        vars_ = vars_.unsqueeze(1).unsqueeze(-1) + self.Hyvarinen_eps       # (mb_size, 1, num_mix_online, 1)
        weights = weights.unsqueeze(1).unsqueeze(-1)  # (mb_size, 1, num_mix_online, 1)

        # Gaussian PDF
        x_centered = x - means                               # (mb_size, num_mix_target, num_mix_online, B)
        normalizer = 1.0 / torch.sqrt(2 * torch.pi * vars_)
        exp_term = torch.exp(-0.5 * (x_centered ** 2) / vars_)
        gauss_vals = normalizer * exp_term                   # (mb_size, num_mix_target, num_mix_online, B)

        # Compute Q(x)
        Q = torch.sum(weights * gauss_vals, dim=2)           # (mb_size, num_mix_target, B)

        # Compute Q'(x)
        Q_prime_num = torch.sum(
            weights * (-x_centered / vars_) * gauss_vals,
            dim=2
        )  # (mb_size, num_mix_target, B)

        # Compute Q''(x)
        term1 = ((x_centered ** 2) / (vars_ ** 2)) - (1.0 / vars_)  # (mb_size, num_mix_target, num_mix_online, B)
        Q_double_prime_num = torch.sum(
            weights * term1 * gauss_vals,
            dim=2
        )  # (mb_size, num_mix_target, B)

        # Hyvarinen score: Q'' / Q - 0.5 * (Q'/Q)^2
        S = Q_double_prime_num / Q - 0.5 * (Q_prime_num / Q) ** 2  # (mb_size, num_mix_target, B)
        S_avg = S.mean(dim=-1)  # (mb_size, num_mix_target)
        loss_per_sample = torch.sum(target_weights * S_avg, dim=1)  # (mb_size,)
        hyvarinen_loss = loss_per_sample.mean()

        return hyvarinen_loss
    

    def TVD(self, transitions, target_net):   

        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        new_obses = [t[3] for t in transitions]

        obses = np.stack([o.get_frames() for o in obses])
        new_obses = np.stack([o.get_frames() for o in new_obses])

        obses_t = torch.as_tensor(obses, dtype=self.float_type, device=self.device)                         # (mb_size, 4, 84, 84)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)           # (mb_size, 1)
        rews_t = torch.as_tensor(rews, dtype=self.float_type, device=self.device).unsqueeze(-1)             # (mb_size, 1)
        new_obses_t = torch.as_tensor(new_obses, dtype=self.float_type, device=self.device)                 # (mb_size, 4, 84, 84)

        new_actions = self.optimal_policy.act(new_obses_t, epsilon=self.epsilon_target, dtype=self.float_type)
        new_actions_t = torch.as_tensor(new_actions, dtype=torch.int64, device=self.device).unsqueeze(-1) # (mb_size, 1)
        mb_size = obses_t.shape[0]

        # ---- Step 1: Forward pass ----
        output = self.net(obses_t)  # (mb_size, num_actions * num_mix * 3)
        output = output.view(mb_size, self.num_actions, self.num_mixture, 3) # (mb_size, num_actions, num_mix, 3)

        # ---- Step 3: Select by action ----
        action_indices = actions_t.unsqueeze(-1).unsqueeze(-1)  # shape: (mb_size, 1, 1, 1)
        action_indices = action_indices.expand(-1, 1, self.num_mixture, 3)  # (mb_size, 1, num_mix, 3)
        selected = output.gather(1, action_indices).squeeze(1)  # (mb_size, num_mix, 3)
        # print(selected.shape)
 
        # ---- Step 4: Extract and process parameters ----
        log_weights = selected[..., 0]  # (mb_size, num_mix)
        # log_weights = log_weights - log_weights.max() # Prevent exploding to inf value when exponentiated.
        means = selected[..., 1]        # (mb_size, num_mix)
        log_vars = selected[..., 2]     # (mb_size, num_mix)

        vars_ = torch.exp(log_vars)
        # weights = torch.exp(log_weights)
        # weights = weights / torch.sum(weights, dim=1, keepdim=True)
        weights = torch.softmax(log_weights, dim=-1)


        # ---- Step 5: Get target net values (for s', a') ----

        with torch.no_grad():
            target_out = target_net(new_obses_t)
            target_out = target_out.view(mb_size, self.num_actions, self.num_mixture, 3)

            new_action_indices = new_actions_t.unsqueeze(-1).unsqueeze(-1)  # shape: (mb_size, 1, 1, 1)
            new_action_indices = new_action_indices.expand(-1, 1, self.num_mixture, 3)  # (mb_size, 1, num_mix-target, 3)
            target_selected = target_out.gather(1, new_action_indices).squeeze(1)  # (mb_size, num_mix-target, 3)
            # print(target_selected.shape)

            target_log_weights = target_selected[..., 0]
            # target_log_weights = target_log_weights - target_log_weights.max() # Prevent blowing up to infinity when exponentiated.
            target_means = target_selected[..., 1]
            target_log_vars = target_selected[..., 2]

            target_vars = torch.exp(target_log_vars)
            # target_weights = torch.exp(target_log_weights)
            # target_weights = target_weights / torch.sum(target_weights, dim=1, keepdim=True)
            target_weights = torch.softmax(target_log_weights, dim=-1)


        # ---- Step 6: Compute the TVD Loss (resample-approximated) ----

        # means   # (mb_size, num_mix-online)
        # vars_   # (mb_size, num_mix-online)
        # weights # (mb_size, num_mix-online)
        means_target = rews_t + self.GAMMA * target_means  # (mb_size, num_mix-target)
        vars_target = (self.GAMMA ** 2) * target_vars      # (mb_size, num_mix-target)
        # target_weights                                   # (mb_size, num_mix-target)

        ## Sample x from target policy
        means_target_exp = means_target.unsqueeze(2).expand(mb_size, self.num_mixture, self.TVD_resample_mixture) # (mb_size, num_mix-target, B_samples)
        vars_target_exp = vars_target.unsqueeze(2).expand(mb_size, self.num_mixture, self.TVD_resample_mixture) + self.TVD_eps   # (mb_size, num_mix-target, B_samples)
        std_target = torch.sqrt(vars_target_exp)         # (mb_size, num_mix-target, B_samples)
        normal_samples = torch.randn_like(std_target)    # (mb_size, num_mix-target, B_samples)
        x_samples = means_target_exp + std_target * normal_samples  # (mb_size, num_mix_target, B)


        # Expand for broadcasting
        x = x_samples.unsqueeze(2)                                    # (mb_size, num_mix_target, 1, B_samples)
        means = means.unsqueeze(1).unsqueeze(-1)                      # (mb_size, 1, num_mix_online, 1)
        vars_ = vars_.unsqueeze(1).unsqueeze(-1) + self.TVD_eps       # (mb_size, 1, num_mix_online, 1)
        weights = weights.unsqueeze(1).unsqueeze(-1)                  # (mb_size, 1, num_mix_online, 1)

        # Evaluate online GMM density at x
        norm_online = 1.0 / torch.sqrt(2 * torch.pi * vars_)
        exp_online = torch.exp(-0.5 * ((x - means) ** 2) / vars_)
        p_online = torch.sum(weights * norm_online * exp_online, dim=2)  # (mb_size, num_mix_target, B_samples)

        # Target network GMM evaluated at x_samples
        normalizer_target = 1.0 / torch.sqrt(2 * torch.pi * vars_target_exp)  # (mb_size, num_mix_target, B_samples)
        exp_term_target = torch.exp(-0.5 * ((x_samples - means_target_exp) ** 2) / vars_target_exp)  # (mb_size, num_mix_target, B_samples)
        phi_target = normalizer_target * exp_term_target  # (mb_size, num_mix_target, B_samples)
        p_target = torch.sum(target_weights.unsqueeze(-1) * phi_target, dim=1)  # (mb_size, B_samples)
        p_target = p_target.unsqueeze(1) + self.TVD_eps  # (mb_size, 1, B_samples) for broadcasting

        # Ratio and error term
        ratio = p_online / p_target  # (mb_size, num_mix_target, B_samples)
        error = torch.abs(1.0 - ratio)  # (mb_size, num_mix_target, B_samples)
        L1_indiv = error.mean(dim=-1)  # (mb_size, num_mix_target)
        weighted_error = torch.sum(L1_indiv * target_weights, dim=1)  # (mb_size,)
        TVD_loss = weighted_error.mean()

        return TVD_loss
    

def cnn_IQN(observation_space, depths=(32,64,64)): # IQN has its own NN-structure.
    n_input_channels = observation_space.shape[0] # observation_space.shape: (4, 84, 84) in breakout.
    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4), # (mb, 4, 84, 84) -> (mb, 32, 20, 20)
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2), # (mb, 32, 20, 20) -> (mb, 64, 9, 9)
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1), # (mb, 64, 9, 9) -> (mb, 64, 7, 7)
        nn.ReLU(),
        nn.Flatten() # (mb, 64*7*7)
    )

    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1] # no need to convert to cuda.

    return cnn, n_flatten



class Particle_OPE(nn.Module):
    def __init__(self, env, optimal_policy, epsilon_target, device, method, num_particle, float_type, GAMMA, sigma_laplace, sigma_rbf, kappa):
        super().__init__()

        self.num_actions = env.action_space.n
        self.optimal_policy = optimal_policy
        self.device = device 

        self.num_mixture = num_particle
        self.float_type = float_type
        self.GAMMA = GAMMA
        self.epsilon_target = epsilon_target

        self.sigma_laplace = sigma_laplace
        self.sigma_rbf = sigma_rbf
 
        self.method = method
        self.kappa = kappa

        # if method=="Energy":
        #     self.K0=self.K0_energy
        # elif method=="Laplace":
        #     self.K0=self.K0_laplace
        # elif method=="RBF":
        #     self.K0=self.K0_rbf


        ## Third trial (include quantile-approach)
        if self.method=="IQN":
            conv_net, middle_dim = cnn_IQN(env.observation_space)
            self.huberloss=torch.nn.HuberLoss(reduction='none', delta=self.kappa)
            self.n_cos_trans = self.num_mixture

            ## psi (CNN): image (mb, 4, 84, 84) -> (mb, 64*7*7)
            self.conv_net = conv_net

            ## phi: cos-tranformed (mb, n_cos_trans) -> (mb, 64*7*7)
            self.middle_dim = middle_dim        
            self.phi = nn.Sequential(nn.Linear(self.n_cos_trans, self.middle_dim), nn.ReLU()) # (mb, n_cos_trans) -> (mb, 64*7*7)

            ## final: (mb, 64*7*7) -> (mb, num_actions*n_quant)
            # self.finals = nn.Sequential(nn.Linear(self.middle_dim, 512), nn.ReLU(), nn.Linear(512, self.num_actions))   
            self.finals = nn.Sequential(
                nn.Linear(self.middle_dim, 512), 
                nn.ReLU(), 
                nn.Linear(512, 450),
                nn.ReLU(),
                nn.Linear(450, 400),
                nn.ReLU(),
                nn.Linear(400, 350),
                nn.ReLU(),
                nn.Linear(350, 300),
                nn.ReLU(),
                nn.Linear(300, 250),
                nn.ReLU(),
                nn.Linear(250, 200),
                nn.ReLU(),
                nn.Linear(200, 150),
                nn.ReLU(),
                nn.Linear(150, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_actions)   
            )
        else:
            conv_net = nature_cnn(env.observation_space)
            self.huberloss=torch.nn.HuberLoss(reduction='none', delta=self.kappa)
            self.fc = nn.Sequential(
                nn.Linear(512, 450),
                nn.ReLU(),
                nn.Linear(450, 400),
                nn.ReLU(),
                nn.Linear(400, 350),
                nn.ReLU(),
                nn.Linear(350, 300),
                nn.ReLU(),
                nn.Linear(300, 250),
                nn.ReLU(),
                nn.Linear(250, 200),
                nn.ReLU(),
                nn.Linear(200, 150),
                nn.ReLU(),
                nn.Linear(150, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_actions * self.num_mixture)   
            )
            self.net = nn.Sequential(conv_net, self.fc)

    def forward(self, x):
        if self.method=="IQN":
            psi_x = self.conv_net(x) # (mb, 64*7*7)
            tau = torch.rand(self.num_mixture , 1).to(device=self.device, dtype=self.float_type) # (n_quant, 1)
            quants = torch.arange(0, self.n_cos_trans, 1.0).to(device=self.device, dtype=self.float_type).unsqueeze(0) # (1, n_cos_trans)
            cos_trans = torch.cos(tau * quants * np.pi) # (n_quant, n_cos_trans)
            phi_tau = self.phi(cos_trans) # (n_quant, 64*7*7)

            psi_x = psi_x.unsqueeze(1) # (mb, 1, 64*7*7) # reorganize for broadcasting.
            phi_tau = phi_tau.unsqueeze(0) # (1, n_quant, 64*7*7)
            interaction = psi_x * phi_tau # (mb, n_quant, 64*7*7) 

            F_values = self.finals(interaction) # (mb, n_quant, num_actions)
            action_values = F_values.transpose(1, 2) # (m, num_actions, n_quant)
            return action_values, tau # (m, num_actions, n_quant), (n_quant, 1)
        else:
            return self.net(x)

    def behavior_act(self, obses, epsilon_behavior): # behavior policy

        actions=self.optimal_policy.act(obses, epsilon=epsilon_behavior, dtype=self.float_type)
        return actions

    def QRDQN(self, transitions, target_net):

        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        new_obses = [t[3] for t in transitions]

        obses = np.stack([o.get_frames() for o in obses])
        new_obses = np.stack([o.get_frames() for o in new_obses])

        obses_t = torch.as_tensor(obses, dtype=self.float_type, device=self.device)                         # (mb_size, 4, 84, 84)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)           # (mb_size, 1)
        rews_t = torch.as_tensor(rews, dtype=self.float_type, device=self.device).unsqueeze(-1)             # (mb_size, 1)
        new_obses_t = torch.as_tensor(new_obses, dtype=self.float_type, device=self.device)                 # (mb_size, 4, 84, 84)

        new_actions = self.optimal_policy.act(new_obses_t, epsilon=self.epsilon_target, dtype=self.float_type)
        # new_actions_t = torch.as_tensor(new_actions, dtype=torch.int64, device=self.device).unsqueeze(-1) # (mb_size, 1)
        mb_size = obses_t.shape[0]

        eval_q = self.net(obses_t)  # (mb_size, num_actions * num_mix)
        # print(eval_q.shape)
        tau = np.arange(1,self.num_mixture+1) / (self.num_mixture+1)
        eval_q = eval_q.reshape(mb_size, self.num_actions, self.num_mixture)  # batch * action * quantiles
        eval_q = torch.stack([eval_q[i][actions_t[i]] for i in range(mb_size)]).squeeze(1) # batch * quantiles
        # print(eval_q.shape)

        with torch.no_grad():
            next_q=target_net(new_obses_t)
            next_q=next_q.reshape(mb_size, self.num_actions, self.num_mixture)
            next_q=torch.stack([next_q[i][new_actions[i]] for i in range(mb_size)]) # batch * quantiles
            target_q=rews_t + self.GAMMA * next_q # batch * quantiles

        eval_q = eval_q.unsqueeze(2)
        target_q = target_q.unsqueeze(1)
        # print(target_q.requires_grad)

        u_values = target_q - eval_q # batch x quantiles x quantiles

        tau_values = torch.as_tensor(tau, dtype=self.float_type, device=self.device).view(1,-1,1) # 1 x n_quant x 1        
        # print(tau_values.shape)

        weight = torch.abs(tau_values - u_values.le(0).float()) # mb_size x n_quant x n_quant # Logical values should be switched into float. 
        # print(weight.shape)

        rho_values = self.huberloss(eval_q, target_q) # mb_size x n_quant x n_quant (sample by eval by target)
        # print(rho_values.shape)

        loss_bybatch = (weight*rho_values).mean(dim=2).sum(dim=1) # mean over target, sum over eval. => mb_size 
        loss = loss_bybatch.mean() # mean over samples

        return loss 

    def IQN(self, transitions, target_net):

        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        # dones = np.asarray([t[3] for t in transitions])
        new_obses = [t[3] for t in transitions]

        # if isinstance(obses[0], PytorchLazyFrames):
        obses = np.stack([o.get_frames() for o in obses])
        new_obses = np.stack([o.get_frames() for o in new_obses])
        # else:
        #     obses = np.asarray([obses])
        #     new_obses = np.asarray([new_obses])

        obses_t = torch.as_tensor(obses, dtype=self.float_type, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1) # needs to be a column vector
        rews_t = torch.as_tensor(rews, dtype=self.float_type, device=self.device).unsqueeze(-1)     # needs to be a column vector
        # dones_t = torch.as_tensor(dones, dtype=self.float_type, device=self.device).unsqueeze(-1)   # needs to be a column vector
        new_obses_t = torch.as_tensor(new_obses, dtype=self.float_type, device=self.device)


        ## Compute evaluation distribution
        eval_q_distribution, tau_eval = self(obses_t) # (mb_size, num_actions, n_quant), (n_quant, 1)
        mb_size = eval_q_distribution.size(0)
        eval_q_dist = torch.stack([eval_q_distribution[i][actions_t[i]] for i in range(mb_size)]).squeeze(1) # Z_online(s_t,a_t) : fixed action => (mb_size, n_quant: eval)

        ## Compute next state distribution
        next_q_distribution, tau_target = target_net(new_obses_t)
        next_q_distribution = next_q_distribution.detach() # (mb_size, num_actions, n_quant: target)

        # next_q_values = next_q_distribution.mean(dim=2) # (mb_size, num_actions)
        # best_actions = torch.argmax(next_q_values, dim=1) # (mb_size,)

        new_actions = self.optimal_policy.act(new_obses_t, epsilon=self.epsilon_target, dtype=self.float_type)
        next_q_dist = torch.stack([next_q_distribution[i][new_actions[i]] for i in range(mb_size)]) # (mb_size, n_quant: target)

        target_q_dist = rews_t + self.GAMMA * next_q_dist # (mb_size, n_quant: target)


        ## Compute loss => Objective: Huber loss (Dabney)
        eval_q_dist = eval_q_dist.unsqueeze(2) # (mb_size, n_quant: eval, 1) # unsqueeze in torch = expand_dims in numpy
        target_q_dist = target_q_dist.unsqueeze(1) # (mb_size, 1, n_quant: target)
        u_values = target_q_dist.detach() - eval_q_dist # may detach # (mb_size, n_quant: eval, n_quant: target)
        tau_values = torch.as_tensor(tau_eval, dtype=self.float_type, device=self.device).view(1,-1,1) # (1, n_quant : eval, 1)        
        weight = torch.abs(tau_values - u_values.le(0).float()) # (mb_size, n_quant: eval, n_quant: target)
    
        rho_values = self.huberloss(eval_q_dist, target_q_dist.detach()) / self.kappa # (mb_size, n_quant: eval, n_quant: target) => may detach
        loss_bybatch = (weight*rho_values).mean(dim=2).sum(dim=1) # (mb_size,) mean over target, sum over eval. => mb_size 
        loss = loss_bybatch.mean() # mean over samples

        return loss
        






    