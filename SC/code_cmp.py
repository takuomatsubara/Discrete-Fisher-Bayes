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
from Posteriors import FDBayes, KSDBayes, Bayes
from Models import CMP



#==========================================================================
# Parse options
#==========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--type', default="FDBayes", type=str)
parser.add_argument('--numboot', default=100, type=int)
args = parser.parse_args()



#==========================================================================
# Set hyper-seeds
#==========================================================================

torch.manual_seed(0)



#==================================================
# Instantiate: Model and Posterior
#==================================================

model = CMP()

prior = torch.distributions.Poisson(torch.tensor([3.0, 3.0]), validate_args=False)
log_prior = lambda param: prior.log_prob(param).sum()
transit_p = torch.distributions.Normal(torch.zeros(2), 0.1*torch.ones(2))

File_ID = args.type + "_nboot=" + str(args.numboot)

if args.type == "FDBayes":
    labeltitle="DFD Bayes"
    posterior = FDBayes(model.ratio_m, model.ratio_p, model.stat_m, model.stat_p, log_prior)
elif args.type == "KSDBayes":
    labeltitle="KSD Bayes"
    posterior = KSDBayes(model.ratio_m, model.stat_m, model.shift_p, log_prior)
elif args.type == "Bayes":
    posterior = Bayes(model.uloglikelihood, torch.arange(100).reshape(100, 1), log_prior)
elif args.type == "ExchangeMCMC":
    posterior = ExchangeMCMC(model.sample, model.uloglikelihood, log_prior)
elif args.type == "MMDBayes":
    posterior = MMDBayes(model.sample, log_prior)
elif args.type == "ABC":
    def feature(X):
        SX = model.stat_m(X) / X.shape[1]
        mean = torch.mean(SX)
        return torch.Tensor([mean])
    loss = lambda F1, F2: torch.sqrt(torch.mean((F1 - F2)**2))
    posterior = ABC(model.sample, feature, loss, prior, args.epsilon)
    File_ID += '_epsilon=' + str(args.epsilon)
else:
    posterior = None
    
print(File_ID)



#==================================================
# Execute
#==================================================

data = torch.Tensor(pd.read_csv('./Data/Sales.csv').values)
data = data - 1

posterior.set_X(data)

p_init, _ = posterior.minimise(posterior.loss, prior.sample(), ite=50000, lr=0.1, loss_thin=100, progress=False)
boot_minimisers, _ = posterior.bootstrap_minimisers(data, args.numboot, lambda: p_init)
posterior.set_X(data)
beta_opt = posterior.optimal_beta(posterior.loss, boot_minimisers)
post_sample = posterior.sample(5000, 5000, transit_p, prior.sample(), beta=beta_opt)

np.save("./Res/"+File_ID+"_samples_beta="+str(float(beta_opt))+".npy", post_sample)



#==================================================
# Predictive
#==================================================

num = 33

bin_idx = np.arange(num-1)
label_0 = np.array(["Data"]*(num-1))
label_1 = np.array([labeltitle]*(num-1))
    
hist = np.histogram(data.numpy().squeeze(), bins=np.arange(num))[0]
df_np = np.c_[label_0, bin_idx, hist]
    
for ith in range(0, post_sample.shape[0], 100):
    pred = model.sample(post_sample[ith,:], 100*data.shape[0], 10*data.shape[0])[::100,]
    hist = np.histogram(pred.numpy().squeeze(), bins=np.arange(num))[0]
    df_np_ith = np.c_[label_1, bin_idx, hist]
    df_np = np.r_[df_np, df_np_ith]
    
df = pd.DataFrame(data=df_np, columns=['label', 'index', 'count'])
df = df.astype({"label": "category", "index": int, "count": int})
df.to_pickle("./Res/"+File_ID+"_preds.pkl")


