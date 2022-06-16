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
parser.add_argument('--theta2', default=0.75s, type=float)
parser.add_argument('--dnum', default=2000, type=int)
parser.add_argument('--pnum', default=5000, type=int)
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
data = cmp.sample(torch.tensor([args.theta1, args.theta2]), 100*args.dnum, 10*args.dnum)[::100,]

prior = torch.distributions.Chi2(torch.tensor([3.0, 3.0]))
log_prior = lambda param: prior.log_prob(param).sum()
transit_p = torch.distributions.Normal(torch.zeros(2), 0.1*torch.ones(2))

File_ID = args.type + '_theta1=' + str(args.theta1) + '_theta2=' + str(args.theta2) + '_numboot=' + str(args.numboot) + '_dnum=' + str(args.dnum) + '_pnum=' + str(args.pnum)

if args.type == "FDBayes":
    posterior = FDBayes(cmp.ratio_m, cmp.ratio_p, cmp.stat_m, cmp.stat_p, log_prior)
elif args.type == "KSDBayes":
    posterior = KSDBayes(cmp.ratio_m, cmp.stat_m, cmp.shift_p, log_prior)
elif args.type == "Bayes":
    posterior = Bayes(cmp.uloglikelihood, torch.arange(100).reshape(100, 1), log_prior)
else:
    posterior = None

print(File_ID)



#==================================================
# Execute
#==================================================

posterior.set_X(data)

Ps = np.zeros((10, args.pnum, 2))
beta_opt = torch.tensor([1.0])

if args.type == "FDBayes" or args.type == "KSDBayes":
    p_init, _ = posterior.minimise(posterior.loss, prior.sample(), ite=50000, lr=0.1, loss_thin=100, progress=False)
    boot_minimisers, _ = posterior.bootstrap_minimisers(data, args.numboot, lambda: p_init)
    posterior.set_X(data)
    beta_opt = posterior.optimal_beta(posterior.loss, boot_minimisers)
    np.save("./Res/"+File_ID+"_bootminimisers_beta="+str(float(beta_opt))+".npy", boot_minimisers.numpy())
    
for ith in range(10):
    post_sample = posterior.sample(args.pnum, args.pnum, transit_p, prior.sample(), beta=beta_opt)
    Ps[ith, :, :] = post_sample.numpy()
np.save("./Res/"+File_ID+"_samples.npy", Ps)


