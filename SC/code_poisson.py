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



#==========================================================================
# Parse options
#==========================================================================

parser = argparse.ArgumentParser()
args = parser.parse_args()

File_ID = 'Bayes_Poisson'
print(File_ID)



#==========================================================================
# Set hyper-seeds
#==========================================================================

torch.manual_seed(0)



#==================================================
# Define: Posterior 
#==================================================

class Bayes():
    
    def __init__(self, log_prior):
        super(Bayes, self).__init__()
        self.log_prior = log_prior
        self.unif = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))

        
    def set_X(self, X):
        self.X = X
        
        
    def neglogl(self, param):
        return - ( self.X * np.log(param[0]) - param[0] - torch.lgamma(self.X+1) ).sum()
    
    
    def log_potential(self, param, beta=1.0):
        return - beta * self.neglogl(param) + self.log_prior(param)

    
    def sample(self, num_sample, num_burnin, transit_p, state_init, beta=1.0):
        state_old = state_init
        chain = []
        
        for i in tqdm(range(num_burnin + num_sample)):
            move = transit_p.sample()
            state_new = ( state_old.log() + move ).exp() # exp for change of variable R->[0,\infty)
            
            log_p_old = self.log_potential(state_old, beta) + state_old.log().sum()
            log_p_new = self.log_potential(state_new, beta) + state_new.log().sum()
            
            if self.unif.sample().log() < min(0, log_p_new - log_p_old):
                state_old = state_new
            chain.append(state_old.tolist())
            
        return torch.tensor(chain[num_burnin:])



#==================================================
# Data
#==================================================

data = torch.Tensor(pd.read_csv('./Data/Sales.csv').values)
data = data - 1



#==================================================
# Instantiate: Model and Posterior
#==================================================

prior = torch.distributions.Poisson(torch.tensor([3.0]), validate_args=False)
bayes = Bayes(lambda param: prior.log_prob(param).sum())
transit_p = torch.distributions.Normal(torch.zeros(1), 0.1*torch.ones(1))

bayes.set_X(data)
post_sample = bayes.sample(5000, 5000, transit_p, prior.sample())
np.save("./Res/"+File_ID+"_samples.npy", post_sample)



#==================================================
# Generate Predictive
#==================================================

num = 33

bin_idx = np.arange(num-1)
label_0 = np.array(["Data"]*(num-1))
label_1 = np.array(["Bayes"]*(num-1))

hist = np.histogram(data.numpy().squeeze(), bins=np.arange(num))[0]
df_np = np.c_[label_0, bin_idx, hist]

for ith in range(0, post_sample.shape[0], 100):
    dist = torch.distributions.Poisson(post_sample[ith,:])
    pred = dist.sample(sample_shape=(data.shape[0],))
    hist = np.histogram(pred.numpy().squeeze(), bins=np.arange(num))[0]
    df_np_ith = np.c_[label_1, bin_idx, hist]
    df_np = np.r_[df_np, df_np_ith]

df = pd.DataFrame(data=df_np, columns=['label', 'index', 'count'])
df = df.astype({"label": "category", "index": int, "count": int})
df.to_pickle("./Res/"+File_ID+"_preds.pkl")


