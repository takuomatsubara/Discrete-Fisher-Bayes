# Discrete-Fisher-Bayes

This repository contains [Pytorch](https://pytorch.org/) codes to reproduce all the experiments in

> [*Generalised Bayesian Inference for Discrete Intractable Likelihoods.*](https://arxiv.org/abs/2206.08420)
> T. Matsubara, J. Knoblauch, F. Briol, C. J. Oates.
>
> Discrete state spaces represent a major computational challenge to statistical inference, since the computation of normalisation constants requires summation over large or possibly infinite sets, which can be impractical. This paper addresses this computational challenge through the development of a novel generalised Bayesian inference procedure suitable for discrete intractable likelihood. Inspired by recent methodological advances for continuous data, the main idea is to update beliefs about model parameters using a discrete Fisher divergence, in lieu of the problematic intractable likelihood. The result is a generalised posterior that can be sampled using standard computational tools, such as Markov chain Monte Carlo, circumventing the intractable normalising constant. The statistical properties of the generalised posterior are analysed, with sufficient conditions for posterior consistency and asymptotic normality established. In addition, a novel and general approach to calibration of generalised posteriors is proposed. Applications are presented on lattice models for discrete spatial data and on multivariate models for count data, where in each case the methodology facilitates generalised Bayesian inference at low computational cost.

The following four experiments were considered in the paper: 

1. Simulated data fitting by the Conway-Maxwell-Poisson model
2. Sales data modelling by the Conway-Maxwell-Poisson model
3. Simualted data fitting by the Ising model
4. Gene expression data modelling by the Poisson and the Conway-Maxwell-Poisson graphical model

Results for each experiment can be reproduced by running the *.py files in each directory, respectively, CMP, SC, Ising, and Graphical. 



### Data

The experiment 2 and 4 above used the following publicly avaialble datasets:

1. Sales dataset collected by Shmueli et al. (2005) [1]: Shmueli et al. (2005) [1] collected sales figures for a particular item of clothing, taken across the different stores of a large national retailer, where their data can be downloaded from <https://www.stat.cmu.edu/COM-Poisson/Sales-data.html>. We store the processed data in .csv formate in SC/Data.
2. Gene expression dartaset studied in Inouye et al. (2017) [2]: Inouye et al. (2017) [2] preprocessed a subset of data gathered by the Cancer Genome Atlas Program relevant to breast cancer for their analysis, where their data can be downloaded from <https://github.com/davidinouye/sqr-graphical-models>. We store the processed data in .npy format in Graphical/Data.



### Main Classes

The first main class is a high-level class "Posterior" in Source/model.py, which is equipped with a sampling algorithm by the standard Metropolis-Hastings algorithm and our proposed calibration algorithm. One may define a custom posterior as an inherited class of Posterior by overwriting the "loss" attribute depending on a model and a loss to use. The quantities required for our calibration alrogithm can be semi-automatically computed by the automatic differentiation of Pytorch. The documentation is also provided in the source code.

    class Posterior():
    """A high-level class to perform sampling and calibration of a posterior.

    Parameters
    ----------
    log_prior : function
        A function to return the value at each parameter of a log prior that users specify.

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


The second main class is "FDBayes" contined in Source/model.py, which is an inherited class of Posterior. 
It performs inference by the Discrete Fisher Divergence Bayes posterior without access to a normalising constant of a probability model on any discrete space. The documentation is also provided in the source code.

    class FDBayes(Posterior):
    """A class that corresponds to the FDBayes posterior.

    Parameters
    ----------
    ratio_m : function
        The first part in the discrete Fishder divergence that depends on a ratio of a probability model.
    ratio_p : function
        THe second part in the discrete Fishder divergence that depends on a ratio of a probability model. 
    stat_m : function
        A data transform function that applies to an argument of the ratio_m function.
    stat_p : function
        A data transform function that applies to an argument of the ratio_p function.
    log_prior : function
        A function to return the value at each parameter of a log prior that users specify.

    Main Attributes
    ----------
    set_X : function
        One call this function to apply the stat_m and stat_p functions to data before sampling form the posterior.
        The set_X function can be used to perform computations that does not have to be repeated during the sampling process.
        
    loss : function
        A loss corresponds to the discrete Fisher divergence.
        This function overwrites one in the Posterior class.
    
    Notes
    ----------
    Examples of how to specify the parameter of the FDBayes class can be found in each model definition in Source/model.py
    """



### Packages

The source code uses Python 3.10.9 and the following main packages:

1. numpy (version 1.23.5)
2. torch (version 1.12.1)
3. pyro (version 1.8.4)
4. scipy (version 1.10.0)
5. pandas (version 1.5.3)
6. sklearn (version 1.2.1)
7. matplotlib (version 3.7.0)
8. seaborn (version 0.12.2)
9. tqdm (version 4.64.1)



### Other Dependencies

The "Ising/_dependency" folder contains routines for sampling from the Ising model provided in <https://github.com/jiaseny/kdsd>; see [3].



### Reference

1. *G. Shmueli, T. P. Minka, J. B. Kadane, S. Borle, and P. Boatwright. A useful distribution for fitting discrete data: revival of the Conway–Maxwell–Poisson distribution. Journal of the Royal Statistical Society: Series C (Applied Statistics), 54(1):127–142, 2005.*
2. *D. I. Inouye, E. Yang, G. I. Allen, and P. Ravikumar. A review of multivariate distributions for count data derived from the Poisson distribution. Wiley Interdisciplinary Reviews: Computational Statistics, 9(3):e1398, 2017.*
3. *Yang, J., Liu, Q., Rao, V. &amp; Neville, J.. (2018). Goodness-of-Fit Testing for Discrete Distributions via Stein Discrepancy. Proceedings of the 35th International Conference on Machine Learning*


