#==========================================================================
# Import: Libraries
#==========================================================================

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
# Define: Conway-Maxwell-Poisson Model
#==========================================================================

class CMP():
    
    def __init__(self):
        super(CMP, self).__init__()
        
        self.initp = torch.distributions.Poisson(2)
        self.tranp = torch.distributions.Bernoulli(0.5)
        self.unifp = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        
        
    def sample(self, param, num_sample, num_burnin):        
        state_old = self.initp.sample().float()
        chain = torch.zeros(num_burnin + num_sample, 1).float()
        
        for i in range(num_burnin + num_sample):
            move = ( 2 * self.tranp.sample() - 1 ).float()
            state_new = state_old + move
        
            l_new = self._elikelihood(state_new, param)
            l_old = self._elikelihood(state_old, param)
            
            if self.unifp.sample() < min(1, l_new/l_old):
                state_old = state_new
            chain[i] = state_old.abs()
        
        return chain[num_burnin:,]
    
    
    def _elikelihood(self, x, param):
        adj = (x != 0).float()
        return ( param[0] ** x.abs() / torch.lgamma(x.abs()+1).exp()**param[1] ) / (2**adj)
    
    
    def uloglikelihood(self, param, X):
        return X * torch.log(param[0]) - param[1] * torch.lgamma(X+1)
    
    
    def stat_m(self, X):
        #Need to separate X!=0 and X==0 for proper automatic differentiation
        #Concatenate the two cases by the original order
        idx1 = (X.flatten() != 0).nonzero()
        idx0 = (X.flatten() == 0).nonzero()
        idx = torch.cat((idx1, idx0), 0).flatten()
        _, reorder_indices = torch.sort(idx)
        return X[X!=0], X[X==0], reorder_indices
    
        
    def stat_p(self, X):
        return X + 1
    
    
    def ratio_m(self, param, SX_m):
        #Need to separate X!=0 and X==0 for proper automatic differentiation
        #Concatenate the two cases by the original order
        return torch.cat((SX_m[0] ** param[1] / param[0], SX_m[1]))[SX_m[2]].unsqueeze(-1)
    
    
    def ratio_p(self, param, SX_p):
        return SX_p ** param[1] / param[0]
    
    
    def shift_p(self, X):
        return ( X + 1 ).unsqueeze(0)

    

#==========================================================================
# Define: 2D Ising Model
#==========================================================================

class Ising2D():
    def __init__(self, size):
        super(Ising2D, self).__init__()
        
        from ising import Ising
        
        self.size = size
        self.P_mat = self._generate_edge_mat(self.size)
        self.sample_model = Ising(self.size**2)
    
    
    def sample(self, param, num_sample, *args):
        self.sample_model.set_ferromagnet(self.size, float(param))
        return torch.Tensor(self.sample_model.sample(num_iters=1e5, num_samples=num_sample))
    
    
    def _generate_edge_mat(self, dim):
        padmat = torch.zeros(dim, dim, dim, dim)
        for i in range(dim):
            for j in range(dim):
                if not i - 1 == -1:
                    padmat[i][j][i-1, j] = 1
                if not j - 1 == -1:
                    padmat[i][j][i, j-1] = 1
                if not i + 1 == dim:
                    padmat[i][j][i+1, j] = 1
                if not j + 1 == dim:
                    padmat[i][j][i, j+1] = 1
        return padmat.reshape(dim*dim, dim*dim)
    
    
    def stat_m(self, X):
        return - 2 * X * ( X @ self.P_mat )
        
        
    def stat_p(self, X):
        return 2 * X * ( X @ self.P_mat )
        
        
    def ratio_m(self, param, SX_m):
        return torch.exp( SX_m / param )
    
    
    def ratio_p(self, param, SX_p):
        return torch.exp( SX_p / param )
    
    
    def uloglikelihood(self, param, X):
        return torch.sum( X * ( X @ self.P_mat ) ) / param
    
    
    def shift_p(self, X, state=torch.tensor([-1.0,1.0]), diff=2.0):
        X_shift = ( X.unsqueeze(-1) + torch.eye(X.shape[1])*diff ).transpose(1, 2)
        return torch.where((X_shift==state[-1]+diff), state[0], X_shift).transpose(0, 1)
    
    
    def pseudologlikelihood(self, param, X):
        tmp_a = torch.exp( - X @ self.P_mat / param )
        tmp_b = torch.exp( X @ self.P_mat / param )
        tmp_Xa = tmp_a * ( ( X - 1 ).abs() / 2 )
        tmp_Xb = tmp_b * ( ( X + 1 ).abs() / 2 )
        tmp = ( tmp_Xa + tmp_Xb ) / ( tmp_a + tmp_b ) 
        return torch.sum( tmp.log() , axis = 1 )


    
#==========================================================================
# Define: Poisson Graphical Model
#==========================================================================

class PGM():
    def __init__(self, dim):
        super(PGM, self).__init__()
        self.dim = dim
        self.tri = torch.triu_indices(dim, dim)
        self.dim_range = range(dim)
        self.dim_total = dim + int(dim*(dim-1)/2)
        
        self.diag_idx = (torch.eye(dim).bool())[self.tri[0], self.tri[1]]
        self.offd_idx = (~torch.eye(dim).bool())[self.tri[0], self.tri[1]]
    
        self.initp = torch.distributions.Poisson(2*torch.ones(self.dim))
        self.unifp = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        self.jumpsize = 3
    
    
    def sample(self, param, num_sample, num_burnin):  
        state_old = self.initp.sample().float()
        chain = torch.zeros(num_burnin + num_sample, self.dim).float()
        
        for i in tqdm(range(num_burnin + num_sample)):
            move = ( 2.0 * torch.randint(0, 2, (self.dim,)) - 1.0 ) * torch.randint(1, self.jumpsize+1, (self.dim,))
            state_new = state_old + move
            
            logl_new = self._uloglikelihood(state_new, param)
            logl_old = self._uloglikelihood(state_old, param)
            
            if self.unifp.sample().log() < min(0, logl_new - logl_old):
                state_old = state_new
            chain[i] = state_old.abs()
            
        return chain[num_burnin:,]
    
    
    def _uloglikelihood(self, x, param):
        ad = 2 ** ( (x != 0).float().sum() )
        xa = x.abs()
        xt = xa.unsqueeze(-1)
        xx = xt @ xt.t()
        xx[self.dim_range, self.dim_range] = xa
        t0 = xx[self.tri[0], self.tri[1]]
        t1 = t0[self.diag_idx] @ param[self.diag_idx]
        t2 = t0[self.offd_idx] @ ( param[self.offd_idx] ** 2 )
        t3 = torch.lgamma(xa+1).sum()
        return t1 - t2 - t3 - torch.log(ad)
        
        
    def stat_x(self, X):
        F1 = torch.zeros(X.shape[0], self.dim_total)
        F2 = torch.zeros(X.shape[0], self.dim)
        
        for i in range(X.shape[0]):
            xi = X[i,:]
            fxi1 = xi.unsqueeze(-1) @ xi.unsqueeze(0)
            fxi1[self.dim_range, self.dim_range] = xi
            fxi2 = torch.lgamma(xi+1)
            F1[i,:] = fxi1[self.tri[0], self.tri[1]]
            F2[i,:] = fxi2
    
        return F1, F2
    
    
    def stat_m(self, X):        
        F = self.stat_x(X)
        
        F1m = torch.zeros(self.dim, X.shape[0], self.dim_total)
        F2m = torch.zeros(self.dim, X.shape[0], self.dim)
        
        for i in range(X.shape[0]):
            xi = X[i,:]
            for j in range(X.shape[1]):
                yj = xi.clone()
                yj[j] = yj[j] - 1 if yj[j] != 0 else 0
                fyj1 = yj.unsqueeze(-1) @ yj.unsqueeze(0)
                fyj1[self.dim_range, self.dim_range] = yj
                fyj2 = torch.lgamma(yj+1)
                F1m[j,i,:] = fyj1[self.tri[0], self.tri[1]]
                F2m[j,i,:] = fyj2
        
        R1 = F1m - F[0]
        R2 = F2m - F[1]
        return R1[:,:,self.diag_idx], R1[:,:,self.offd_idx], R2
    
    
    def stat_p(self, X):
        F = self.stat_x(X)
        
        F1p = torch.zeros(self.dim, X.shape[0], self.dim_total)
        F2p = torch.zeros(self.dim, X.shape[0], self.dim)
    
        for i in range(X.shape[0]):
            xi = X[i,:]
            for j in range(X.shape[1]):
                yj = xi.clone()
                yj[j] += 1
                fyj1 = yj.unsqueeze(-1) @ yj.unsqueeze(0)
                fyj1[self.dim_range, self.dim_range] = yj
                fyj2 = torch.lgamma(yj+1)
                F1p[j,i,:] = fyj1[self.tri[0], self.tri[1]]
                F2p[j,i,:] = fyj2
    
        R1 = F[0] - F1p
        R2 = F[1] - F2p
        return R1[:,:,self.diag_idx], R1[:,:,self.offd_idx], R2
       
    
    def ratio_m(self, param, SX_m):
        T1 = SX_m[0] @ param[self.diag_idx]
        T2 = SX_m[1] @ ( param[self.offd_idx] ** 2 )
        T3 = SX_m[2].sum(axis=2)
        return torch.exp( T1 - T2 - T3 ).t()
    
    
    def ratio_p(self, param, SX_p):
        T1 = SX_p[0] @ param[self.diag_idx]
        T2 = SX_p[1] @ ( param[self.offd_idx] ** 2 )
        T3 = SX_p[2].sum(axis=2)
        return torch.exp( T1 - T2 - T3 ).t()
    

    
#==========================================================================
# Define: Conway-Maxwell-Poisson Graphical Model
#==========================================================================

class CMPGM():
    def __init__(self, dim):
        super(CMPGM, self).__init__()
        self.dim = dim
        self.tri = torch.triu_indices(dim, dim)
        self.dim_range = range(dim)
        self.dim_total = dim + int(dim*(dim-1)/2) + dim
        
        self.diag_idx_0 = (torch.eye(dim).bool())[self.tri[0], self.tri[1]]
        self.offd_idx_0 = (~torch.eye(dim).bool())[self.tri[0], self.tri[1]]
        self.addt_idx_0 = torch.zeros(dim).bool()
        self.diag_idx = torch.cat((self.diag_idx_0, self.addt_idx_0))
        self.offd_idx = torch.cat((self.offd_idx_0, self.addt_idx_0))
        self.addt_idx = torch.cat(((self.diag_idx_0*False), (~self.addt_idx_0)))
        
        self.initp = torch.distributions.Poisson(2*torch.ones(self.dim))
        self.unifp = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        self.jumpsize = 3
        
        
    def sample(self, param, num_sample, num_burnin):  
        state_old = self.initp.sample().float()
        chain = torch.zeros(num_burnin + num_sample, self.dim).float()
        
        for i in tqdm(range(num_burnin + num_sample)):
            move = ( 2.0 * torch.randint(0, 2, (self.dim,)) - 1.0 ) * torch.randint(1, self.jumpsize+1, (self.dim,))
            state_new = state_old + move
            
            logl_new = self._uloglikelihood(state_new, param)
            logl_old = self._uloglikelihood(state_old, param)
            
            if self.unifp.sample().log() < min(0, logl_new - logl_old):
                state_old = state_new
            chain[i] = state_old.abs()
            
        return chain[num_burnin:,]
    
    
    def _uloglikelihood(self, x, param):
        ad = 2 ** ( (x != 0).float().sum() )
        xa = x.abs()
        xt = xa.unsqueeze(-1)
        xx = xt @ xt.t()
        xx[self.dim_range, self.dim_range] = xa
        t0 = xx[self.tri[0], self.tri[1]]
        t1 = t0[self.diag_idx_0] @ param[self.diag_idx]
        t2 = t0[self.offd_idx_0] @ ( param[self.offd_idx] ** 2 )
        t3 = torch.lgamma(xa+1) @ ( param[self.addt_idx] ** 2 )
        return t1 - t2 - t3 - torch.log(ad)
    
    
    def stat_x(self, X):
        F1 = torch.zeros(X.shape[0], self.dim_total - self.dim)
        F2 = torch.zeros(X.shape[0], self.dim)
    
        for i in range(X.shape[0]):
            xi = X[i,:]
            fxi1 = xi.unsqueeze(-1) @ xi.unsqueeze(0)
            fxi1[self.dim_range, self.dim_range] = xi
            fxi2 = torch.lgamma(xi+1)
            F1[i,:] = fxi1[self.tri[0], self.tri[1]]
            F2[i,:] = fxi2
    
        return F1, F2
    
    
    def stat_m(self, X):
        F = self.stat_x(X)
        
        diag_idx = (torch.eye(self.dim).bool())[self.tri[0], self.tri[1]]
        offd_idx = (~torch.eye(self.dim).bool())[self.tri[0], self.tri[1]]
        
        F1m = torch.zeros(self.dim, X.shape[0], self.dim_total - self.dim)
        F2m = torch.zeros(self.dim, X.shape[0], self.dim)
        
        for i in range(X.shape[0]):
            xi = X[i,:]
            for j in range(X.shape[1]):
                yj = xi.clone()
                yj[j] = yj[j] - 1 if yj[j] != 0 else 0
                fyj1 = yj.unsqueeze(-1) @ yj.unsqueeze(0)
                fyj1[self.dim_range, self.dim_range] = yj
                fyj2 = torch.lgamma(yj+1)
                F1m[j,i,:] = fyj1[self.tri[0], self.tri[1]]
                F2m[j,i,:] = fyj2
        
        R1 = F1m - F[0]
        R2 = F2m - F[1]
        return R1[:,:,diag_idx], R1[:,:,offd_idx], R2
    
    
    def stat_p(self, X):
        F = self.stat_x(X)
        
        diag_idx = (torch.eye(self.dim).bool())[self.tri[0], self.tri[1]]
        offd_idx = (~torch.eye(self.dim).bool())[self.tri[0], self.tri[1]]
        
        F1p = torch.zeros(self.dim, X.shape[0], self.dim_total - self.dim)
        F2p = torch.zeros(self.dim, X.shape[0], self.dim)
        
        for i in range(X.shape[0]):
            xi = X[i,:]
            for j in range(X.shape[1]):
                yj = xi.clone()
                yj[j] += 1
                fyj1 = yj.unsqueeze(-1) @ yj.unsqueeze(0)
                fyj1[self.dim_range, self.dim_range] = yj
                fyj2 = torch.lgamma(yj+1)
                F1p[j,i,:] = fyj1[self.tri[0], self.tri[1]]
                F2p[j,i,:] = fyj2
    
        R1 = F[0] - F1p
        R2 = F[1] - F2p
        return R1[:,:,diag_idx], R1[:,:,offd_idx], R2
    
    
    def ratio_m(self, param, SX_m):
        T1 = SX_m[0] @ param[self.diag_idx]
        T2 = SX_m[1] @ ( param[self.offd_idx] ** 2 )
        T3 = SX_m[2] @ ( param[self.addt_idx] ** 2 )
        return torch.exp( T1 - T2 - T3 ).t()
    
    
    def ratio_p(self, param, SX_p):
        T1 = SX_p[0] @ param[self.diag_idx]
        T2 = SX_p[1] @ ( param[self.offd_idx] ** 2 )
        T3 = SX_p[2] @ ( param[self.addt_idx] ** 2 )
        return torch.exp( T1 - T2 - T3 ).t()
    
    
    
    