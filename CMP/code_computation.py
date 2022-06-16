#==================================================
# Library Import
#==================================================

import math
import time
import argparse
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

import torch
import torch.autograd as autograd

import sys
sys.path.insert(1, '../Source/')
from Posteriors import FDBayes, KSDBayes, Bayes
from Models import CMP



#==========================================================================
# Parse options
#==========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--type', default="FDBayes", type=str)
parser.add_argument('--theta1', default=4.00, type=float)
parser.add_argument('--theta2', default=0.75, type=float)
parser.add_argument('--numboot', default=100, type=int)
args = parser.parse_args()



#==========================================================================
# Set hyper-seeds
#==========================================================================

torch.manual_seed(0)



#==================================================
# Instantiate: Model and Posterior
#==================================================

cmp = CMP()

prior = torch.distributions.Chi2(torch.tensor([3.0, 3.0]))
log_prior = lambda param: prior.log_prob(param).sum()
transit_p = torch.distributions.Normal(torch.zeros(2), 0.1*torch.ones(2))

File_ID = 'ComputationTime_' + args.type + '_theta1=' + str(args.theta1) + '_theta2=' + str(args.theta2) + '_numboot=' + str(args.numboot)

if args.type == "FDBayes":
    posterior = FDBayes(cmp.ratio_m, cmp.ratio_p, cmp.stat_m, cmp.stat_p, log_prior)
elif args.type == "KSDBayes":
    posterior = KSDBayes(cmp.ratio_m, cmp.stat_m, cmp.shift_p, log_prior)
elif args.type == "Bayes":
    posterior = Bayes(cmp.uloglikelihood, torch.arange(500).reshape(500, 1), log_prior)
else:
    posterior = None
    
print(File_ID)



#==================================================
# Execute
#==================================================

Ns = [500, 1000, 1500, 2000, 2500, 3000]
Ps = np.zeros((len(Ns), 10, 5000, 2))
Ts_P = np.zeros((len(Ns), 10))

NM = 10000
p0 = prior.sample()
full_data = cmp.sample(torch.tensor([args.theta1, args.theta2]), 10*NM*100, 20000)[::100,:]

datas = []
for ith in range(len(Ns)):
    datas.append([])
    for jth in range(10):
        idxj = torch.randint(10*NM, (Ns[ith],))
        datas[ith].append(full_data[idxj,:])

for ith in range(len(Ns)):
    for jth in range(10):
        data = datas[ith][jth]
        posterior.set_X(data)
        
        beta_opt = torch.tensor([1.0])
        
        #
        '''
        if args.type == "FDBayes" or args.type == "KSDBayes":
            time_start = time.time()
            p_init, _ = posterior.minimise(posterior.loss, p0, ite=50000, lr=0.1, loss_thin=100, progress=False)
            time_end = time.time()
            Ts_B0[ith, jth] = time_end - time_start
            
            time_start = time.time()
            boot_minimisers, _ = posterior.bootstrap_minimisers(data, args.numboot, lambda: p_init)
            posterior.set_X(data)
            beta_opt = posterior.optimal_beta(posterior.loss, boot_minimisers)
            time_end = time.time()
            Ts_B[ith, jth] = time_end - time_start
        '''
        
        time_start = time.time()
        post_sample_beta = posterior.sample(5000, 5000, transit_p, prior.sample(), beta=beta_opt)
        time_end = time.time()
        Ts_P[ith, jth] = time_end - time_start
        Ps[ith, jth, :, :] = post_sample_beta.numpy()

np.save("./Res/"+File_ID+"_samples.npy", Ps)
np.save("./Res/"+File_ID+"_times.npy", Ts_P)
#np.save("./Res/"+File_ID+"_times.npy", {'P': Ts_P, 'B': Ts_B, 'B0': Ts_B0})


