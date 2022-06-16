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
import pyro.ops.stats as pyrostats

import sys
sys.path.insert(1, '../Source/')
from Posteriors import FDBayes
from Models import CMPGM



#==========================================================================
# Parse options
#==========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--num_dim', default=10, type=int)
parser.add_argument('--num_boot', default=100, type=int)
args = parser.parse_args()



#==========================================================================
# Set hyper-seeds
#==========================================================================

torch.manual_seed(0)



#==================================================
# Instantiate: Model and Posterior
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
# Define: Model
#==================================================

File_ID = "CMPGM_FDBayes"
posterior = FDBayes(model.ratio_m, model.ratio_p, model.stat_m, model.stat_p, log_prior)



#==================================================
# FDBayes Beta Lyddon et al 2018
#==================================================

def loss_func(param, posterior):
    Ratio_M = posterior.ratio_m(param, posterior.SX_m)
    Ratio_P = posterior.ratio_p(param, posterior.SX_p)
    return ( Ratio_M**2 - 2*Ratio_P ).sum(axis=1)


def get_beta_lyddon(p_ast, data, posterior):
    loss_each = lambda param: loss_func(param, posterior)
    S = torch.autograd.functional.jacobian(loss_each, p_ast)
    I = S.t() @ S / data.shape[0]
    J = torch.autograd.functional.hessian(posterior.loss, p_ast) / data.shape[0]
    try:
        I_inv = torch.pinverse(I)
    except:
        return 100000
    return torch.trace( J @ I_inv @ J.t() ) / torch.trace( J )



#==================================================
# Execute
#==================================================

Beta_opt = torch.zeros(10)
Beta_lyd = torch.zeros(10)

p0 = param_init()

for ith in range(10):
    bootidx = torch.randint(data.shape[0], (data.shape[0],))
    dat = data[bootidx,:]
    
    posterior.set_X(dat)
    p_init, error = posterior.minimise(posterior.loss, p0, ite=50000, lr=0.001, loss_thin=100, progress=False)
    
    boot_minimisers, boot_losses = posterior.bootstrap_minimisers(dat, args.num_boot, lambda: p_init, ite=10000, lr=0.001)
    posterior.set_X(dat)
    Beta_opt[ith] = posterior.optimal_beta(posterior.loss, boot_minimisers)
    Beta_lyd[ith] = get_beta_lyddon(p_init, dat, posterior)

np.save("./Res/"+File_ID+"_beta_opt.npy", Beta_opt.numpy())
np.save("./Res/"+File_ID+"_beta_lyd.npy", Beta_lyd.numpy())


