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
from Posteriors import FDBayes, KSDBayes, PseudoBayes
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

File_ID = args.type + '_size=' + str(args.size) + '_theta=' + str(args.theta) + '_dnum=' + str(args.dnum) + '_pnum=' + str(args.pnum)

if args.type == "FDBayes":
    posterior = FDBayes(model.ratio_m, model.ratio_p, model.stat_m, model.stat_p, log_prior)
elif args.type == "KSDBayes":
    posterior = KSDBayes(model.ratio_m, model.stat_m, model.shift_p, log_prior)
elif args.type == "PseudoBayes":
    posterior = PseudoBayes(model.pseudologlikelihood, log_prior)
else:
    posterior = None
    
print(File_ID)



#==================================================
# Execute
#==================================================

post_samples = torch.zeros(10, args.pnum)
times_post = torch.zeros(10)
time0_beta = 0.0
betas = torch.zeros(10)
times_beta = torch.zeros(10)

p0 = prior.sample()

data = model.sample(torch.tensor([args.theta]), args.dnum)
posterior.set_X(data)
p_init, _ = posterior.minimise(posterior.loss, p0, ite=5000, lr=0.01, loss_thin=100, progress=False)
    
time_start_1 = time.time()
boot_minimisers, _ = posterior.bootstrap_minimisers(data, args.numboot, lambda: p_init, lr=0.01)
posterior.set_X(data)
beta_opt = posterior.optimal_beta(posterior.loss, boot_minimisers)
time_end_1 = time.time()
time0_beta = time_end_1 - time_start_1

for i in range(10):
    time_start_2 = time.time()
    post_sample = posterior.sample(args.pnum, args.pnum, transit_p, prior.sample(), beta=beta_opt).flatten()
    time_end_2 = time.time()
    
    post_samples[i] = post_sample
    times_post[i] = time_end_2 - time_start_2

np.save("./Res/"+File_ID+"_samples.npy", post_samples.numpy())


for i in range(10):
    data = model.sample(torch.tensor([args.theta]), args.dnum)
    posterior.set_X(data)
    p_init, _ = posterior.minimise(posterior.loss, p0, ite=5000, lr=0.01, loss_thin=100, progress=False)
    
    time_start_1 = time.time()
    boot_minimisers, _ = posterior.bootstrap_minimisers(data, args.numboot, lambda: p_init, lr=0.01)
    posterior.set_X(data)
    beta_opt = posterior.optimal_beta(posterior.loss, boot_minimisers)
    time_end_1 = time.time()
    
    betas[i] = beta_opt
    times_beta[i] = time_end_1 - time_start_1
    
    
np.save("./Res/"+File_ID+"_betas.npy", betas.numpy())
np.save("./Res/"+File_ID+"_times.npy", {'post':times_post.numpy(), 'beta0': time0_beta, 'beta':times_beta.numpy()})


