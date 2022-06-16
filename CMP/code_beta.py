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

File_ID = 'CompareBeta_' + args.type + '_theta1=' + str(args.theta1) + '_theta2=' + str(args.theta2) + '_nboot=' + str(args.numboot)

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
# FDBayes Beta Lyddon et al 2018
#==================================================

def loss_func_0(x, param):
    c1 = 0
    c2 = 1 / param[0]
    return c1**2 - 2 * c2


def loss_func_x(x, param):
    c1 = x ** param[1] / param[0]
    c2 = (x + 1) ** param[1] / param[0]
    return c1**2 - 2 * c2


def get_beta_lyddon(p_ast, data):
    I = torch.zeros(2, 2)
    J = torch.zeros(2, 2)
    for x in data:
        if x == 0:
            loss = lambda param: loss_func_0(x, param)
        else:
            loss = lambda param: loss_func_x(x, param)
        S = torch.autograd.functional.jacobian(loss, p_ast)
        I += S.t() @ S / data.shape[0]
        J += torch.autograd.functional.hessian(loss, p_ast) / data.shape[0]
    I_inv = torch.inverse(I)
    
    return torch.trace( J @ I_inv @ J.t() ) / torch.trace( J )



#==================================================
# Execute
#==================================================

#Ns = [200, 300, 500, 1000, 2000, 3000, 5000, 10000]
Ns = [500, 1000, 1500, 2000, 2500, 3000]
Bs_O = np.zeros((len(Ns), 10))
Bs_L = np.zeros((len(Ns), 10))

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
        
        p_init, _ = posterior.minimise(posterior.loss, p0, ite=50000, lr=0.1, loss_thin=100, progress=False)
        boot_minimisers, _ = posterior.bootstrap_minimisers(data, args.numboot, lambda: p_init)
        posterior.set_X(data)
        beta_opt = posterior.optimal_beta(posterior.loss, boot_minimisers)
        
        beta_lyd = get_beta_lyddon(p_init, data)
        
        Bs_O[ith, jth] = beta_opt.numpy()
        Bs_L[ith, jth] = beta_lyd.numpy()

np.save("./Res/"+File_ID+"_betas.npy", {'O': Bs_O, 'L': Bs_L})


