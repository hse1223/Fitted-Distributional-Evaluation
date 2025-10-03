##### Setting #####

from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import kv
import os
import pickle
import numpy as np
import time
import argparse


### Argparse

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, choices=['PDFL2', 'Energy', 'RBF', 'Laplace', 'KL', 'FLE'])
parser.add_argument('--N_size', type=int)
parser.add_argument('--setting', type=int, choices=[1,2,3,4], help="gamma, bound_val, reward_sigma, update_thresh")
parser.add_argument('--simulation_num', type=int, default=50)
parser.add_argument('--sigma_laplace', type=float, default=1.0)
parser.add_argument('--gridno_laplace', type=int, default=100)
parser.add_argument('--sigma_rbf', type=float, default=1.0)
parser.add_argument('--cheat_initial', action='store_true', help="whether to allow setting the true next value as the intial value in each iteration.")
args = parser.parse_args()

method = args.method
N_size = args.N_size
setting = args.setting
simulation_num = args.simulation_num
sigma_laplace = args.sigma_laplace
sigma_rbf = args.sigma_rbf
cheat_initial = args.cheat_initial
gridno_laplace = args.gridno_laplace


### Select hyperparameters

num_actions = 5
# alpha = 0.01 # tail probability for pdf approximation.
# if setting==1:
#     gamma = 0.90; bound_val=100; reward_sigma = 1; update_thresh=1e-10; C_divideT=1; Laplace_grid=False
# elif setting==2:
#     gamma = 0.90; bound_val=10000; reward_sigma = 5; update_thresh=1e-10; C_divideT=1; Laplace_grid=False
if setting==1:
    gamma = 0.99; bound_val=10000; reward_sigma = 1; update_thresh=1e-10; C_divideT=10; Laplace_grid=False 
elif setting==2:
    gamma = 0.99; bound_val=10000; reward_sigma = 5; update_thresh=1e-10; C_divideT=10; Laplace_grid=True



def T_choosing(N_sample):
    # p_wasserstein=1
    delta_surrogate=1; l_objective=5; q_inaccuracy=1; c_contraction=1 # Energy - p.33 of draft12
    first = 1 / (c_contraction - 1 / (2*q_inaccuracy))
    second = min(delta_surrogate, 1 / q_inaccuracy) / (2 * (l_objective - 1)) 
    third = np.log(N_sample) / np.log(1/gamma)
    # C_divideT=1
    return max(round(first*second*third / C_divideT ),1)

T=T_choosing(N_size)
n=N_size//T


### Parameters

action_pool = np.linspace(0, 2 * np.pi, num_actions, endpoint=False)

A=np.array([[0.6,0],[0,0.8]]); B=np.array([[0.2,0],[0.,0.1]])
K=np.array([[1.,0],[0,1]]) 
# K=np.array([[np.cos(action_pool[-1]), -np.sin(action_pool[-1])],[np.sin(action_pool[-1]), np.cos(action_pool[-1])]]) 
Q=np.array([[4.,1],[1,4]]); R=np.array([[2.,1],[1,2]]) 
# print(K)


##### Objective function #####

def PDF_L2(theta123vec):

    x_samples = [t[0] for t in transitions]
    a_samples = [t[1] for t in transitions]
    reward_samples = [t[2] for t in transitions]
    xprime_samples = [t[3] for t in transitions]

    M1current = np.array([[theta123vec[0], theta123vec[1]],[theta123vec[2], theta123vec[3]]])
    M2current = np.array([[theta123vec[4], theta123vec[5]],[theta123vec[6], theta123vec[7]]])
    M3current = np.array([[theta123vec[8], theta123vec[9]],[theta123vec[10], theta123vec[11]]])

    LHS_mean = np.array([x_samples[ind].T @ M1current @ x_samples[ind] + a_samples[ind].T @ M2current @ x_samples[ind] + a_samples[ind].T @ M3current @ a_samples[ind] for ind in range(n)])
    RHS_mean = np.array([reward_samples[ind] + xprime_samples[ind].T @ Mtilde @ xprime_samples[ind] for ind in range(n)])
    LHS_sd = reward_sigma * np.sqrt(1 / (1-gamma**2))    
    RHS_sd = reward_sigma * np.sqrt(gamma**2 / (1-gamma**2))    

    LHS_var = LHS_sd**2
    RHS_var = RHS_sd**2

    term1 = 1 / np.sqrt(4 * np.pi * LHS_var)
    term2 = 1 / np.sqrt(4 * np.pi * RHS_var)
    var_sum = LHS_var + RHS_var
    diff_squared = (LHS_mean - RHS_mean)**2
    term3 = (2 / np.sqrt(2 * np.pi * var_sum)) * np.exp(- diff_squared / (2 * var_sum))
    # print(term1); print(term2); print(len(term3))

    values_indiv = term1 + term2 - term3

    return values_indiv.mean()


def MMD_squared(theta123vec):

    x_samples = [t[0] for t in transitions]
    a_samples = [t[1] for t in transitions]
    reward_samples = [t[2] for t in transitions]
    xprime_samples = [t[3] for t in transitions]

    M1current = np.array([[theta123vec[0], theta123vec[1]],[theta123vec[2], theta123vec[3]]])
    M2current = np.array([[theta123vec[4], theta123vec[5]],[theta123vec[6], theta123vec[7]]])
    M3current = np.array([[theta123vec[8], theta123vec[9]],[theta123vec[10], theta123vec[11]]])

    LHS_mean = np.array([x_samples[ind].T @ M1current @ x_samples[ind] + a_samples[ind].T @ M2current @ x_samples[ind] + a_samples[ind].T @ M3current @ a_samples[ind] for ind in range(n)])
    RHS_mean = np.array([reward_samples[ind] + xprime_samples[ind].T @ Mtilde @ xprime_samples[ind] for ind in range(n)])
    LHS_sd = reward_sigma * np.sqrt(1 / (1-gamma**2))    
    RHS_sd = reward_sigma * np.sqrt(gamma**2 / (1-gamma**2))    

    term1_mean = LHS_mean - LHS_mean
    term1_var = LHS_sd**2 + LHS_sd**2
    term2_mean = LHS_mean - RHS_mean
    term2_var = LHS_sd**2 + RHS_sd**2
    # print(len(term1_mean)); print(len(term2_mean)); print(term1_sd); print(term2_sd)

    if method=="Energy":
        K0 = K0_energy
    elif method=="RBF":
        K0 = K0_rbf
    elif method=="Laplace":
        K0 = K0_laplace

    mmd_indiv = K0(term1_mean, term1_var) - 2 * K0(term2_mean, term2_var)
    # print(mmd_indiv); print(mmd_indiv.shape)
    # exit()

    return mmd_indiv.mean()


def K0_energy(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(var)
    abs_mu_over_sigma = np.abs(mu / sigma)
    term1 = sigma * np.sqrt(2 / np.pi) * np.exp(-mu**2 / (2 * var))
    term2 = np.abs(mu) * (1 - 2 * norm.cdf(-abs_mu_over_sigma))
    return - (term1 + term2)

def K0_rbf(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    term1 = 1 / (np.sqrt(2 * np.pi) * sigma_rbf)
    term2 = 1 / np.sqrt(1 + var / (2 * sigma_rbf**2))
    exponent = -mu**2 / (4 * sigma_rbf**2 + 2 * var)
    return term1 * term2 * np.exp(exponent)

def K0_laplace(mu: np.ndarray, var: np.ndarray) -> np.ndarray:

    if not Laplace_grid:
        sigma = np.sqrt(var)
        term1 = np.exp(0.5 * (var / sigma_laplace**2) - mu / sigma_laplace)
        phi1 = norm.cdf(mu / sigma - sigma / sigma_laplace)
        term2 = np.exp(0.5 * (var / sigma_laplace**2) + mu / sigma_laplace)
        phi2 = norm.cdf(-mu / sigma - sigma / sigma_laplace)
        return term1 * phi1 + term2 * phi2
    else:
        sigma = np.sqrt(var)        
        z_min = norm.ppf(0.01)
        z_max = norm.ppf(0.99)
        z_grid = np.linspace(z_min, z_max, gridno_laplace)  # shape: (gridno,)
        dz = z_grid[1] - z_grid[0]        # uniform spacing

        z_grid = z_grid[np.newaxis, ...]  # shape: (1, gridno)
        mu = mu[..., np.newaxis]          # shape: (n, 1)
        y = mu + sigma * z_grid           # shape: (n, gridno)

        k_vals = np.exp(-np.abs(y) / sigma_laplace)
        normal_pdf = norm.pdf(z_grid)     # shape: (1, gridno)
        integrand = k_vals * normal_pdf   # shape: (n, gridno)
        integral = np.sum(integrand, axis=-1) * dz
        return integral



def KL(theta123vec): # non-MC FLE

    x_samples = [t[0] for t in transitions]
    a_samples = [t[1] for t in transitions]
    reward_samples = [t[2] for t in transitions]
    xprime_samples = [t[3] for t in transitions]

    M1current = np.array([[theta123vec[0], theta123vec[1]],[theta123vec[2], theta123vec[3]]])
    M2current = np.array([[theta123vec[4], theta123vec[5]],[theta123vec[6], theta123vec[7]]])
    M3current = np.array([[theta123vec[8], theta123vec[9]],[theta123vec[10], theta123vec[11]]])

    LHS_mean = np.array([x_samples[ind].T @ M1current @ x_samples[ind] + a_samples[ind].T @ M2current @ x_samples[ind] + a_samples[ind].T @ M3current @ a_samples[ind] for ind in range(n)])
    RHS_mean = np.array([reward_samples[ind] + xprime_samples[ind].T @ Mtilde @ xprime_samples[ind] for ind in range(n)])
    LHS_sd = reward_sigma * np.sqrt(1 / (1-gamma**2))    
    RHS_sd = reward_sigma * np.sqrt(gamma**2 / (1-gamma**2))    

    KL_sum = 0
    for index in range(n):

        mu1 = RHS_mean[index]
        sigma1 = RHS_sd
        mu2 = LHS_mean[index]
        sigma2 = LHS_sd        

        KL_indiv = np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
        KL_sum += KL_indiv

    return KL_sum


def NeglogL(theta123vec): # FLE with MC-error

    x_samples = [t[0] for t in transitions]
    a_samples = [t[1] for t in transitions]
    reward_samples = [t[2] for t in transitions]
    xprime_samples = [t[3] for t in transitions]

    M1current = np.array([[theta123vec[0], theta123vec[1]],[theta123vec[2], theta123vec[3]]])
    M2current = np.array([[theta123vec[4], theta123vec[5]],[theta123vec[6], theta123vec[7]]])
    M3current = np.array([[theta123vec[8], theta123vec[9]],[theta123vec[10], theta123vec[11]]])

    LHS_mean = np.array([x_samples[ind].T @ M1current @ x_samples[ind] + a_samples[ind].T @ M2current @ x_samples[ind] + a_samples[ind].T @ M3current @ a_samples[ind] for ind in range(n)])
    RHS_mean = np.array([reward_samples[ind] + xprime_samples[ind].T @ Mtilde @ xprime_samples[ind] for ind in range(n)])
    RHS_sd = reward_sigma * np.sqrt(gamma**2 / (1-gamma**2))    

    RHS_samples = np.random.normal(loc=RHS_mean, scale=RHS_sd, size=n)
    logL_simplified = np.sum((LHS_mean - RHS_samples)**2) # equivalent with minimizing -logL.

    return logL_simplified





##### Simulations ##### 

M1true=np.array([[1.,1.],[1.,1.]])*1; M2true = M1true.copy(); M3true = M1true.copy()
for iteration in range(100000):
    Mtilde_temp = gamma * (M1true + K.T @ M2true + K.T @ M3true @ K)
    M1true = Q + A.T @ Mtilde_temp @ A 
    M2true = B.T @ Mtilde_temp @ A + B.T @ Mtilde_temp.T @ A 
    M3true = R + B.T @ Mtilde_temp @ B 

print("\n\nmethod="+method)
print("Setting="+str(setting))
print("Cheating-initial=" + str(cheat_initial))
print("Nsize="+str(N_size) + " : (T,n)=(" + str(T) + ", " + str(n) + ")")

print("\n\n")
print('True:')
print(M1true); print(M2true); print(M3true)
bounds = [(-bound_val, bound_val)] * 12



### dpi sampling & Inaccuracy function

dpi_filename = 'dpi_samples/setting{}.pkl'.format(setting)
if os.path.exists(dpi_filename):
    print(f"Loading precomputed dpi samples from {dpi_filename}...")
    with open(dpi_filename, 'rb') as f:
        saved_data = pickle.load(f)
        x_samples_dpi = saved_data['x_samples_dpi']
        a_samples_dpi = saved_data['a_samples_dpi']
        M_accuracysim = len(x_samples_dpi)
        # print(M_accuracysim)
else:
    os.makedirs(os.path.dirname("dpi_samples/"), exist_ok=True)   
    print(f"No precomputed file found. Generating dpi samples and saving to {dpi_filename}...")
    np.random.seed(123456789) # ensure the same samples for inaccuracy.
    M_accuracysim = 100000
    # M_accuracysim = 100
    unif_inaccuracy = np.random.uniform(0,1,M_accuracysim)
    theta_samples_inaccuracy = (2*np.pi) * unif_inaccuracy
    radius_samples_inaccuracy = np.random.uniform(0,1,M_accuracysim)
    x1_samples_inaccuracy = radius_samples_inaccuracy * np.cos(theta_samples_inaccuracy)
    x2_samples_inaccuracy = radius_samples_inaccuracy * np.sin(theta_samples_inaccuracy)
    x_samples_rho = np.stack((x1_samples_inaccuracy, x2_samples_inaccuracy)).T   # (M, 2) array
    a_samples_rho = x_samples_rho @ K.T
    # print(x_samples_inaccuracy.shape); print(a_samples_inaccuracy.shape)


    x_samples_dpi = []
    a_samples_dpi = []
    for x_sample_inaccuracy, a_sample_inaccuracy in zip(x_samples_rho, a_samples_rho):
        accept=False

        # iter=0
        while not accept:
            # print(iter)
            # print(x_sample_inaccuracy)
            # print(a_sample_inaccuracy)
            u = np.random.uniform(0, 1)  # Step 2(i): Sample u ~ Uniform(0,1)
            if u <= 1 - gamma:
                accept = True  # Step 2(ii): Accept with probability (1 - γ)
                x_samples_dpi.append(x_sample_inaccuracy)
                a_samples_dpi.append(a_sample_inaccuracy)
            else:
                xprime_sample_inaccuracy = x_sample_inaccuracy @ A.T + a_sample_inaccuracy @ B.T
                aprime_sample_inaccuracy = xprime_sample_inaccuracy @ K.T
                x_sample_inaccuracy = xprime_sample_inaccuracy
                a_sample_inaccuracy = aprime_sample_inaccuracy
                # iter+=1
                # exit()

    x_samples_dpi = np.vstack(x_samples_dpi)
    a_samples_dpi = np.vstack(a_samples_dpi)
    with open(dpi_filename, 'wb') as f:
        pickle.dump({'x_samples_dpi': x_samples_dpi, 'a_samples_dpi': a_samples_dpi}, f)
    print("\nx,a samples (for inaccuracy measurement) generated from dpi.\n")
    # print(x_samples_dpi); print(a_samples_dpi)


def inaccuracy(M1_estimate, M2_estimate, M3_estimate):
    Wasserstein_sum = 0
    for x, a in zip(x_samples_dpi, a_samples_dpi):
        Wasserstein_value = abs(x.T @ (M1_estimate - M1true) @ x  + a.T @ (M2_estimate - M2true) @ x + a.T @ (M3_estimate - M3true) @ a)
        Wasserstein_sum += Wasserstein_value**2
    Wasserstein_approx = np.sqrt(Wasserstein_sum / M_accuracysim)
    return Wasserstein_approx


Wasserstein_approx_list = []
for seedno in range(simulation_num):

    start_time = time.time()

    ### Step-1: Data collection 

    np.random.seed(seedno)

    # N=n*T
    unif = np.random.uniform(0,1,N_size)
    theta_samples = (2*np.pi) * unif  # theta~unif(0,2pi)
    radius_samples = np.random.uniform(0,1,N_size)   # r~unif(0,1)
    x1_samples = radius_samples * np.cos(theta_samples)
    x2_samples = radius_samples * np.sin(theta_samples)
    x_samples = np.stack((x1_samples, x2_samples)).T   # (N,2) array of (x1,x2)

    a_samples = np.random.choice(action_pool, size=N_size, replace=True)
    cos_a = np.cos(a_samples)
    sin_a = np.sin(a_samples)
    rotation_matrices = np.stack([
        np.stack([cos_a, -sin_a], axis=-1),
        np.stack([sin_a, cos_a], axis=-1)
    ], axis=-2)  # Shape: (N,2,2)
    a_samples = np.einsum('nij,nj->ni', rotation_matrices, x_samples)
    # print(np.max(np.abs(np.linalg.norm(x_samples, axis=1)-np.linalg.norm(a_samples, axis=1)))) # confirm ||x||==||a|| for all samples (true rotations).

    xprime_samples = x_samples @ A.T + a_samples @ B.T
    reward_samples = []
    for x_sample, a_sample in zip(x_samples, a_samples):
        print(x_sample.T @ Q @ x_sample + a_sample.T @ R @ a_sample)
        reward_sample = x_sample.T @ Q @ x_sample + a_sample.T @ R @ a_sample + np.random.normal(loc=0,scale=reward_sigma,size=1)[0]
        reward_samples.append(reward_sample)
    # print(x_samples.shape, a_samples.shape, xprime_samples.shape, len(reward_samples))

    transitions_whole = [(x_samples[ind], a_samples[ind], reward_samples[ind], xprime_samples[ind]) for ind in range(N_size)]
    subdatasets = [transitions_whole[i * n:(i + 1) * n] for i in range(T - 1)]
    subdatasets.append(transitions_whole[(T - 1) * n:])
    # print(N_size, T, n)
    # print(len(subdatasets), len(subdatasets[0]), len(subdatasets[T-1])) # verify if correctly splitted.


    ### Step-2: Fitted Iterations 

    inaccuracy_list = []
    M1prev=np.array([[1.,1.],[1.,1.]])*1
    M2prev=np.array([[1.,1.],[1.,1.]])*1
    M3prev=np.array([[1.,1.],[1.,1.]])*1 

    firstreachedcriteria=T
    consecutive_count=0
    for iteration in range(T):

        print('\nMethod=' + method + ': Simulation ' + str(seedno+1) + " : " + str(iteration) + "-th iteration:")

        # print(M1prev); print(M2prev); print(M3prev)

        if iteration <= T // 2 + 1:
            print("W-1,dpi,1-Inaccuracy = {:.20f}".format(inaccuracy(M1prev, M2prev, M3prev)))
        else:
            print("W-1,dpi,1-Inaccuracy = {:.20f}".format(inaccuracy_indiv))

        transitions = subdatasets.pop()

        thetavec1prev=np.array([M1prev[0][0], M1prev[0][1], M1prev[1][0], M1prev[1][1]])
        thetavec2prev=np.array([M2prev[0][0], M2prev[0][1], M2prev[1][0], M2prev[1][1]])
        thetavec3prev=np.array([M3prev[0][0], M3prev[0][1], M3prev[1][0], M3prev[1][1]])
        thetavec123prev = np.concatenate((thetavec1prev, thetavec2prev, thetavec3prev))

        Mtilde = gamma * (M1prev + K.T @ M2prev + K.T @ M3prev @ K) # Even without cheating-initial, necessary for calculating.

        if method=="PDFL2":
            result = minimize(PDF_L2, thetavec123prev, method='L-BFGS-B', bounds=bounds) # reasonable
        elif method=="Energy":
            result = minimize(MMD_squared, thetavec123prev, method='L-BFGS-B', bounds=bounds)
        elif method=="RBF":
            result = minimize(MMD_squared, thetavec123prev, method='L-BFGS-B', bounds=bounds)
        elif method=="Laplace":
            result = minimize(MMD_squared, thetavec123prev, method='L-BFGS-B', bounds=bounds)
        elif method=="KL":
            result = minimize(KL, thetavec123prev, method='L-BFGS-B', bounds=bounds)
        elif method=="FLE":
            result = minimize(NeglogL, thetavec123prev, method='L-BFGS-B', bounds=bounds)


        if cheat_initial:
            M1star = Q + A.T @ Mtilde @ A 
            M2star = B.T @ Mtilde @ A + B.T @ Mtilde.T @ A 
            M3star = R + B.T @ Mtilde @ B 
            thetavec1star=np.array([M1star[0][0], M1star[0][1], M1star[1][0], M1star[1][1]])
            thetavec2star=np.array([M2star[0][0], M2star[0][1], M2star[1][0], M2star[1][1]])
            thetavec3star=np.array([M3star[0][0], M3star[0][1], M3star[1][0], M3star[1][1]])
            thetavec123star = np.concatenate((thetavec1star, thetavec2star, thetavec3star))  # Will be used for initial value in a single iteration.
            thetavec123prev_cheat=thetavec123star

            if method=="PDFL2":
                result_cheat = minimize(PDF_L2, thetavec123prev_cheat, method='L-BFGS-B', bounds=bounds) # reasonable
            elif method=="Energy":
                result_cheat = minimize(MMD_squared, thetavec123prev_cheat, method='L-BFGS-B', bounds=bounds)
            elif method=="RBF":
                result_cheat = minimize(MMD_squared, thetavec123prev_cheat, method='L-BFGS-B', bounds=bounds)
            elif method=="Laplace":
                result_cheat = minimize(MMD_squared, thetavec123prev_cheat, method='L-BFGS-B', bounds=bounds)
            elif method=="KL":
                result_cheat = minimize(KL, thetavec123prev_cheat, method='L-BFGS-B', bounds=bounds)
            elif method=="FLE":
                result_cheat = minimize(NeglogL, thetavec123prev_cheat, method='L-BFGS-B', bounds=bounds)

            if result.fun > result_cheat.fun:
                result = result_cheat


        M1est = np.array([[result.x[0], result.x[1]], [result.x[2], result.x[3]]])
        M2est = np.array([[result.x[4], result.x[5]], [result.x[6], result.x[7]]])
        M3est = np.array([[result.x[8], result.x[9]], [result.x[10], result.x[11]]])

        if np.max(abs(M1est-M1prev)) < update_thresh and np.max(abs(M2est-M2prev)) < update_thresh and np.max(abs(M3est-M3prev)) < update_thresh: # Prevent being stuck in local minimizer.
            M1est += np.random.normal(loc=0,scale=1,size=1)[0] 
            M2est += np.random.normal(loc=0,scale=1,size=1)[0] 
            M3est += np.random.normal(loc=0,scale=1,size=1)[0] 

        M1prev = M1est; M2prev = M2est; M3prev = M3est

        if iteration > T // 2 :
            inaccuracy_indiv = inaccuracy(M1prev, M2prev, M3prev)
            inaccuracy_list.append(inaccuracy_indiv)

    ### Step-3: Inaccuracy

    inaccuracy_final = inaccuracy(M1prev, M2prev, M3prev)
    inaccuracy_trajmean = np.mean(inaccuracy_list)

    print("\nMethod=" + method + ": Simulation-index=" + str(seedno+1))
    print("Estimate:")
    print(M1prev); print(M2prev); print(M3prev)
    print("W1-inaccuracy of final estimate = {:.4f}".format(inaccuracy_final))
    print("W1-inaccuracy mean after T//2 iterations = {:.4f}".format(inaccuracy_trajmean))

    end_time = time.time()
    iteration_time = end_time - start_time
    print(f"Computation time: {iteration_time:.4f} seconds.\n")

    Wasserstein_approx_list.append((inaccuracy_final, inaccuracy_trajmean))


##### Save ##### 

if not cheat_initial:
    directory_name="Results/setting"+str(setting)+"/"
else:
    directory_name="Results/setting"+str(setting)+"_cheatedinitial/"
        

if not os.path.exists(directory_name):
    os.makedirs(directory_name)

filename_wasserstein = directory_name + method + "_gamma" + str(int(gamma*100)) + "Nsize" + str(N_size) + "_Wasserstein" + ".pkl"

with open(filename_wasserstein, "wb") as file:
    pickle.dump(Wasserstein_approx_list, file)



