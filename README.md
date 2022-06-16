# Discrete-Fisher-Bayes

This repository contains [Pytorch](https://pytorch.org/) codes to reproduce the experiments in "Generalised Bayesian Inference for Discrete Intractable Likelihoods".

The first main function is the Discrete Fisher Divergence Bayes posterior, which performs generalised Bayesian inference without access to a normalising constant of a probability model on any discrete space. 

The second main function is the algorithm to automatically calibrate a credible region of any posterior. The source code contains a high-level class "Posterior" equipped with this calibration algorithm. To enjoy the calibration algorithm, one can simply define a custom posterior as a child class from "Posterior" by overwriting the "loss" property. Requirements of the calibration algorithm for the child class are automatically computed by the automatic differentiation of Pytorch.

There are three applications considered in the paper: 

1. The Conway-Maxwell-Poisson model,
2. The Ising model, 
3. The Poisson graphical model / The Conway-Maxwell-Poisson graphical model

Results for each application can be reproduced by running the files *.py.



##### Dependencies

The "Ising/_dependency" folder contains routines for sampling from the Ising model provided in <https://github.com/jiaseny/kdsd>. See:

Yang, J., Liu, Q., Rao, V. &amp; Neville, J.. (2018). Goodness-of-Fit Testing for Discrete Distributions via Stein Discrepancy. Proceedings of the 35th International Conference on Machine Learning

