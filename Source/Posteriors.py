#==========================================================================
# Import: Libraries
#==========================================================================

import numpy as np
from tqdm import tqdm
import torch
from sklearn.model_selection import KFold
import pybobyqa
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc.api import MCMC, NUTS, StreamingMCMC, HMC



#==========================================================================
# Define: Base Posterior 
#==========================================================================

class Posterior():
    """A high level class to perform sampling from and calibration of a posterior.

    Parameters
    ----------
    log_prior : function
        A function to return the value at each parameter of a log prior that user specify.

    Main Attributes
    ----------
    loss : function
        A function to return the value at each parameter of a loss that user specify.
        One can overwrite this function to define one's own posterior.
        One can recover a standard posterior by overwriting this function by log-likelihood.

    sample : function
        One can call this function to get samples from the posterior.

    optimal_beta : function
        A function to return an optimal temperature of the posterior based on the bootstrap loss minimisers users have.

    bootstrap_minimisers : function
        A function to compute the the bootstrap loss minimisers required to compute an optimal temperature.
    """
    
    def __init__(self, log_prior):
        """
        Initialisation of the Posterior class.
        """
        super(Posterior, self).__init__()
        self.log_prior = log_prior
        self.unifp = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        
        
    def set_X(self, X, **kwargs):
        """
        A function that can be called to reduce the computational cost during sampling.
        One can overwirte this function to define one's own process to avoid repetitive computations.
        """
        self.X = X
        
        
    def loss(self, param):
        """
        A function that returns the value of the loss at each parameter.
        One can overwirte this function to define one's own posterior.
        """
        return 0
        
        
    def log_potential(self, param, beta=1.0):
        """
        A function that returns the value of the log of the posterior at each parameter.
        """
        return - beta * self.loss(param) + self.log_prior(param)
    
    
    def sample(self, num_sample, num_burnin, transit_p, state_init, thin=1.0, beta=1.0, domain="positive"):
        """
        Sampling from the posterior by the MH method.

        Parameters
        ----------
        num_sample : the number of sample to obtain.
        num_burnin : the number of burn-in for MCMC.
        transit_p : transition kernel for MCMC (Pytorch Distributions).
        state_init : the initial state of MCMC.
        thin : the number of thinning for MCMC.
        beta : the temperture parameter of the posterior.
        domain : the domain type of the parameters.

        Returns
        -------
        chain : the obtained MCMC chain by the MH method.
        """
        assert num_sample % thin == 0 
        
        if domain == "positive":
            chofv = lambda state: state.log()
            chofv_inv = lambda state: state.exp()
            logl_adjust = lambda state: state.log().sum()
        else:
            chofv = lambda state: state
            chofv_inv = lambda state: state
            logl_adjust = lambda state: 0
        
        state_old = state_init
        chain = torch.zeros(int(num_sample/thin), state_old.shape[0]).float()
        
        for i in tqdm(range(num_burnin)):
            move = transit_p.sample()
            state_new = chofv_inv( chofv(state_old) + move ) 
            
            logl_new = self.log_potential(state_new, beta) + logl_adjust(state_new)
            logl_old = self.log_potential(state_old, beta) + logl_adjust(state_old)
            
            if self.unifp.sample().log() < min(0, logl_new - logl_old):
                state_old = state_new
        
        for i in tqdm(range(num_sample)):
            move = transit_p.sample()
            state_new = chofv_inv( chofv(state_old) + move ) 
            
            logl_new = self.log_potential(state_new, beta) + logl_adjust(state_new)
            logl_old = self.log_potential(state_old, beta) + logl_adjust(state_old)
            
            if self.unifp.sample().log() < min(0, logl_new - logl_old):
                state_old = state_new
            if i % thin == 0:
                chain[int(i/thin)] = state_old
        
        return chain
    
    
    def sample_nuts(self, num_sample, num_burnin, state_init, beta=1.0, num_chains=1):    
        """
        A version of the sample function using the NUTS sampler.
        """
        
        nuts = NUTS(potential_fn = lambda args: - self.log_potential(args['theta'], beta))
        mcmc = MCMC(kernel=nuts, warmup_steps=num_burnin, initial_params={'theta': state_init}, num_samples=num_sample, num_chains=num_chains)
        mcmc.run()
        return mcmc.get_samples()['theta'], mcmc.diagnostics()
    
    
    def optimal_beta(self, loss, params):
        """
        A function to compute an optimal temperature value based on the bootstrap minimisers.
        
        Parameters
        ----------
        loss : the loss function subject to calibration.
        params : the bootstrap minimisers of the loss.

        Returns
        -------
        numer / denom : an optimisal temperature parameter computed by the proposed analytical solution.
        """
        numer = 0
        denom = 0
        
        for param in params:
            dl = torch.autograd.functional.jacobian(loss, param)
            ddl = torch.autograd.functional.hessian(loss, param)
            dp = torch.autograd.functional.jacobian(self.log_prior, param)
            numer += ( dl @ dp + ddl.trace() )
            denom += ( dl.norm(p=2)**2 )
        
        return numer / denom
    
    
    def bootstrap_minimisers(self, data, boot_num, param_init, loss_func=None, ite=5000, lr=0.1, loss_thin=100):
        """
        A function to get minisers of all bootstraped losses.
        
        Parameters
        ----------
        data : the full dataset to use for the calibration process.
        boot_num : the number of bootstrap.
        param_init : the initial state of the parameter of the optimisation part.
        loss_func : the custom loss function if needed, default=None.
        ite : the number of iteration of the optimisation part, default=5000.
        lr : the learning rate of optimisation part.
        loss_thin : the frequency to show the loss value report, default=100.

        Returns
        -------
        boot_minimisers: a set of the minimisers of all bootstraped losses by optimisation.
        boot_losses: a set of the minimum values of all bootstraped losses by optimisation.
        """
        p_init = param_init()
        boot_minimisers = torch.zeros(boot_num, p_init.shape[0])
        boot_losses = torch.zeros(boot_num, int(ite/loss_thin))
        
        pbar = tqdm(range(boot_num))
        for j in pbar:
            p_init = param_init()
            boot_idx = torch.randint(data.shape[0], (data.shape[0],))
            boot_data = data[boot_idx,:]
            
            if loss_func != None:
                loss = lambda param: loss_func(param, boot_data)
            else:
                self.set_X(boot_data)
                loss = self.loss
            
            boot_minimisers[j], boot_losses[j] = self.minimise(loss, p_init, ite=ite, lr=lr, loss_thin=loss_thin)
            pbar.set_description("Final Loss: %.5f" % boot_losses[j,int(ite/loss_thin)-1])
        
        return boot_minimisers, boot_losses
        
    
    def minimise(self, loss, p_init, ite=500, lr=0.1, loss_thin=100, progress=False):     
        """
        An optimisation algorithm used in the bootstrap_minimisers function.
        """
        error = torch.zeros(int(ite/loss_thin))
        param = p_init.detach().clone()
        param.requires_grad = True
        optimizer = torch.optim.Adam([param], lr=lr)
        
        pbar = tqdm(range(ite)) if progress else range(ite)
        for i in pbar:
            l = loss(param)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if i % loss_thin == 0:
                error[int(i/loss_thin)] = l
            pbar.set_description("Loss: %.5f" % l) if progress else None
        
        return param.detach().clone(), error
     

#==========================================================================
# Define: Discrete Fisher Divergence Posterior 
#==========================================================================

class FDBayes(Posterior):
    """A class that corresponds to the FDBayes posterior.

    Parameters
    ----------
    ratio_m : function
        A first part used with squared in the discrete Fishder divergence that depends on a ratio of a probability model.
    ratio_p : function
        A second part used without squared in the discrete Fishder divergence that depends on a ratio of a probability model. 
    stat_m : function
        A data transform function that is used as an argument for the ratio_m function.
    stat_p : function
        A data transform function that is used as an argument for the ratio_p function.
    log_prior : function
        A function to return the value at each parameter of a log prior that user specify.

    Main Attributes
    ----------
    set_X : function
        One call this function to apply the stat_m and stat_p functions to data before sampling form the posterior.
        The use of the set_X function reduces a computational cost as this computation can be performed only once before sampling.
        
    loss : function
        A loss corresponds to the discrete Fisher divergence.
        This function overwrites one in Posterior class.
    
    Notes
    ----------
    Examples of specification of the parameter of the FBBayes class can be found in each model definition in Source/model.py
    """
    
    def __init__(self, ratio_m, ratio_p, stat_m, stat_p, log_prior):
        """
        Initialisation of the FDBayes class.
        """
        super(FDBayes, self).__init__(log_prior)
        self.ratio_m = ratio_m
        self.ratio_p = ratio_p
        self.stat_m = stat_m
        self.stat_p = stat_p
        
        
    def set_X(self, X, **kwargs):
        """
        A function to set the transformed data as an attribute before sampling.
        """
        self.SX_m = self.stat_m(X)
        self.SX_p = self.stat_p(X)
        
        
    def loss(self, param):
        """
        A function to return the value of the loss at each parameter.
        """
        Ratio_M = self.ratio_m(param, self.SX_m)
        Ratio_P = self.ratio_p(param, self.SX_p)
        return ( Ratio_M**2 - 2*Ratio_P ).sum()
    

    
#==========================================================================
# Define: Kernel Stein Discrepancy Posterior 
#==========================================================================

class KSDBayes(Posterior):
    
    def __init__(self, ratio_m, stat_m, shift_p, log_prior):
        super(KSDBayes, self).__init__(log_prior)
        self.ratio_m = ratio_m
        self.stat_m = stat_m
        self.shift_p = shift_p
    
    
    def kernel(self, X1, X2):
        return torch.exp( - torch.cdist(X1, X2, p=0) / X2.shape[1] )
    
    
    def set_X(self, X, **kwargs):
        self.X_num = X.shape[0]
        self.KX = self.kernel(X, X)
        self.KX_D = ( self.kernel(self.shift_p(X), X) - self.KX ).mean(axis=1).t()
        self.SX_m = self.stat_m(X)
        
        
    def loss(self, param):
        Score_M = ( 1 - self.ratio_m(param, self.SX_m) )
        T1 = ( Score_M @ Score_M.t() * self.KX ).mean()
        T2 = ( Score_M * self.KX_D ).sum(axis=1).mean()
        return self.X_num * ( T1 + 2.0 * T2 )
    
    
class RobustKSDBayesIsing(Posterior):
    
    def __init__(self, ratio_m, stat_m, shift_p, log_prior):
        super(RobustKSDBayesIsing, self).__init__(log_prior)
        self.ratio_m = ratio_m
        self.stat_m = stat_m
        self.shift_p = shift_p
    
    
    def kernel(self, X1, X2):
        M1 = ( 1 / ( 1 + torch.exp(-( 90 - torch.sum(X1, dim=-1).abs() )) ) ).unsqueeze(-1)
        M2 = ( 1 / ( 1 + torch.exp(-( 90 - torch.sum(X2, dim=-1).abs() )) ) ).unsqueeze(-1)
        K = torch.exp( - torch.cdist(X1, X2, p=0) / X2.shape[1] )
        return M1 * K * M2.t()
    
    
    def set_X(self, X, **kwargs):
        self.X_num = X.shape[0]
        self.KX = self.kernel(X, X)
        self.KX_D = ( self.kernel(self.shift_p(X), X) - self.KX ).mean(axis=1).t()
        self.SX_m = self.stat_m(X)
        
        
    def loss(self, param):
        Score_M = ( 1 - self.ratio_m(param, self.SX_m) )
        T1 = ( Score_M @ Score_M.t() * self.KX ).mean()
        T2 = ( Score_M * self.KX_D ).sum(axis=1).mean()
        return self.X_num * ( T1 + 2.0 * T2 )
    


#==========================================================================
# Define: Bayes Posterior 
#==========================================================================

class Bayes(Posterior):
    
    def __init__(self, uloglikelihood, domain, log_prior):
        super(Bayes, self).__init__(log_prior)
        self.uloglikelihood = uloglikelihood
        self.domain = domain
        
        
    def set_X(self, X, **kwargs):
        self.X = X
        self.X_num = X.shape[0]
        
        
    def loss(self, param):
        log_Z = self.X_num * self.uloglikelihood(param, self.domain).exp().sum().log()
        return - ( self.uloglikelihood(param, self.X).sum() - log_Z )

    
    
#==========================================================================
# Define: Bayes Posterior Based on Pseudo Log Likelihood
#==========================================================================

class PseudoBayes(Posterior):
    
    def __init__(self, pseudologlikelihood, log_prior):
        super(PseudoBayes, self).__init__(log_prior)
        self.pseudologlikelihood = pseudologlikelihood
        
        
    def set_X(self, X, **kwargs):
        self.X = X
    
    
    def loss(self, param):
        return - self.pseudologlikelihood(param, self.X).sum()

    
