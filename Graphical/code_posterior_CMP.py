#==================================================
# Library Import
#==================================================

import math
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

import torch
import torch.autograd as autograd
import pyro.ops.stats as pyrostats

import os, sys; sys.path.append(os.path.abspath("../Source/"))
from Posteriors import FDBayes
from Models import CMPGM, PGM



#==========================================================================
# Parse options
#==========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--type', default="FDBayes", type=str)
parser.add_argument('--num_dim', default=10, type=int)
parser.add_argument('--num_chains', default=4, type=int)
parser.add_argument('--num_sample', default=10000, type=int)
parser.add_argument('--num_boot', default=100, type=int)
parser.add_argument('--num_pred', default=100, type=int)
parser.add_argument('--num_pred_sample', default=500000, type=int)
args = parser.parse_args()



#==========================================================================
# Set hyper-seeds
#==========================================================================

torch.manual_seed(0)



#==================================================
# Define: Data and Model 
#==================================================

dim = args.num_dim
dim_offd = int(dim*(dim-1)/2)
dim_total = 2 * dim + dim_offd

filename = "brca_" + str(dim)
data = torch.from_numpy(np.load("./Data/"+filename+".npy")).float()
model = CMPGM(dim)



#==================================================
# Define: Prior
#==================================================

prior_1 = torch.distributions.Normal(torch.zeros(dim), 1.0*torch.ones(dim))
prior_2 = torch.distributions.Normal(torch.zeros(dim_offd), (1.0/dim_offd)*torch.ones(dim_offd))
prior_3 = torch.distributions.Normal(torch.zeros(dim), 1.0*torch.ones(dim))


def log_prior(param):
    return prior_1.log_prob(param[model.diag_idx]).sum() \
        + prior_2.log_prob(param[model.offd_idx]).sum() \
        + prior_3.log_prob(param[model.addt_idx]).sum()


def param_init():
    param = torch.zeros(dim_total)
    param[model.diag_idx] = prior_1.sample()
    param[model.offd_idx] = prior_2.sample().abs()
    param[model.addt_idx] = prior_3.sample().abs()
    return param



#==================================================
# Define: Posterior
#==================================================

File_ID = "CMPGM_" + args.type + "_numdim=" + str(args.num_dim) + "_numchains=" + str(args.num_chains) + "_numsample=" + str(args.num_sample) + "_numboot=" + str(args.num_boot) + "_numpred=" + str(args.num_pred) + "_numpredsample=" + str(args.num_pred_sample)

if args.type == "FDBayes":
    posterior = FDBayes(model.ratio_m, model.ratio_p, model.stat_m, model.stat_p, log_prior)
elif args.type == "KSDBayes":
    posterior = KSDBayes(model.ratio_m, model.stat_m, model.shift_p, log_prior)
else:
    posterior = None
    
print(File_ID)



#==================================================
# Execute
#==================================================

posterior.set_X(data)


p_init, _ = posterior.minimise(posterior.loss, param_init(), ite=50000, lr=0.001, loss_thin=100, progress=False)
time_beta_start = time.time()
boot_minimisers, _ = posterior.bootstrap_minimisers(data, args.num_boot, lambda: p_init, ite=10000, lr=0.001)
posterior.set_X(data)
beta_opt = posterior.optimal_beta(posterior.loss, boot_minimisers)
time_beta_end = time.time()
time_beta = time_beta_end - time_beta_start


times = np.zeros(args.num_chains)
post_sample = torch.zeros(args.num_chains, args.num_sample, dim_total)
for i in range(args.num_chains):
    time_start = time.time()
    post_sample[i,:], _ = posterior.sample_nuts(args.num_sample, args.num_sample, param_init(), beta=beta_opt)
    time_end = time.time()
    times[i] = time_end - time_start

np.save("./Res/"+File_ID+"_samples.npy", post_sample.numpy())
np.save("./Res/"+File_ID+"_times.npy", {'beta':beta_opt, 'time_beta':time_beta, 'time_post':times})


post_sample_0 = post_sample[0]
pred_sample = torch.zeros(args.num_pred, data.shape[0], dim)
pred_thin = int( args.num_pred_sample / data.shape[0] ) + 1
for i in range(args.num_pred):
    p0_i = post_sample_0[i*int(args.num_sample/args.num_pred)]
    pred = model.sample(p0_i, args.num_pred_sample, args.num_pred_sample)
    pred_sample[i,:] = pred[::pred_thin,:]

np.save("./Res/"+File_ID+"_predictives.npy", pred_sample.numpy())


