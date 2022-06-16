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

import os, sys; sys.path.append(os.path.abspath("../Source/"))
import os, sys; sys.path.append(os.path.abspath("./_dependency/"))
from Posteriors import FDBayes, RobustKSDBayesIsing, PseudoBayes
from Models import Ising2D



#==========================================================================
# Parse options
#==========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--type', default="FDBayes", type=str)
parser.add_argument('--size', default=10, type=int)
parser.add_argument('--theta', default=5.0, type=float)
parser.add_argument('--dnum', default=1000, type=int)
parser.add_argument('--pnum', default=2000, type=int)
parser.add_argument('--onum', default=100, type=int)
parser.add_argument('--numboot', default=100, type=int)
args = parser.parse_args()



#==========================================================================
# Set hyper-seeds
#==========================================================================

torch.manual_seed(0)
np.random.seed(0)



#==================================================
# Instantiate: Model and Posterior
#==================================================

model = Ising2D(args.size)

prior = torch.distributions.Chi2(torch.tensor([3.0]))
log_prior = lambda param: prior.log_prob(param).sum()
transit_p = torch.distributions.Normal(torch.zeros(1), 0.1*torch.ones(1))

File_ID = 'Robust_' + args.type + '_size=' + str(args.size) + '_theta=' + str(args.theta) + '_dnum=' + str(args.dnum) + '_pnum=' + str(args.pnum)

if args.type == "FDBayes":
    posterior = FDBayes(model.ratio_m, model.ratio_p, model.stat_m, model.stat_p, log_prior)
elif args.type == "RobustKSDBayes":
    posterior = RobustKSDBayesIsing(model.ratio_m, model.stat_m, model.shift_p, log_prior)
elif args.type == "PseudoBayes":
    posterior = PseudoBayes(model.pseudologlikelihood, log_prior)
else:
    posterior = None
    
print(File_ID)



#==================================================
# Execute
#==================================================

post_samples = torch.zeros(10, args.pnum)
p0 = prior.sample()
full_data = model.sample(torch.tensor([args.theta]), args.dnum)

data = full_data.clone()
oidx = torch.randint(args.dnum, (args.onum,))
data[oidx,:] = torch.ones(args.onum, args.size**2)

posterior.set_X(data)
p_init, _ = posterior.minimise(posterior.loss, p0, ite=5000, lr=0.01, loss_thin=100, progress=False)

time_start_1 = time.time()
boot_minimisers, _ = posterior.bootstrap_minimisers(data, args.numboot, lambda: p_init, lr=0.01)
posterior.set_X(data)
beta_opt = posterior.optimal_beta(posterior.loss, boot_minimisers)
time_end_1 = time.time()
time0_beta = time_end_1 - time_start_1

for i in range(10):
    post_sample = posterior.sample(args.pnum, args.pnum, transit_p, prior.sample(), beta=beta_opt).flatten()
    post_samples[i] = post_sample

np.save("./Res/"+File_ID+"_samples_noise.npy", post_samples.numpy())


data = full_data.clone()

posterior.set_X(data)
p_init, _ = posterior.minimise(posterior.loss, p0, ite=5000, lr=0.01, loss_thin=100, progress=False)

time_start_1 = time.time()
boot_minimisers, _ = posterior.bootstrap_minimisers(data, args.numboot, lambda: p_init, lr=0.01)
posterior.set_X(data)
beta_opt = posterior.optimal_beta(posterior.loss, boot_minimisers)
time_end_1 = time.time()
time0_beta = time_end_1 - time_start_1

for i in range(10):
    post_sample = posterior.sample(args.pnum, args.pnum, transit_p, prior.sample(), beta=beta_opt).flatten()
    post_samples[i] = post_sample

np.save("./Res/"+File_ID+"_samples_nonoise.npy", post_samples.numpy())


