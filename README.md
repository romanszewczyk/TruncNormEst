# TruncNormEst

Materials connected with he paper:

Roman Szewczyk, Tadeusz Szumiata, Michał Nowicki
"Symbolic regression-based, phenomenological identification of standard deviation estimators of a symmetrically truncated normal distribution"

This library presents a phenomenological method combining massive Monte Carlo simulations with symbolic regression to determine optimal standard deviation estimators for normal distribution and truncated normal distribution. The study successfully identifies a novel linear fractional function for normal distributions and a second-degree rational function for symmetrically truncated distributions. As a result, it addresses estimation problems where analytical methods are ineffective, particularly for small sample sizes. It is demonstrated that this approach effectively negates systematic bias, which can reach 25% in previously described truncated estimators. These findings are of great importance for high-precision metrology in digital systems. 

**The most important outcome:**

std_opti.m  - final version of optimised standard deviation estimator for normal distribution

std_bnr_opt.m - final version of optimised standard deviation estimator for **truncated** normal distribution
