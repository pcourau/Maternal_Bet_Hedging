import numpy as np
from scipy.stats import multivariate_normal as multinom
from scipy.stats import norm
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
from seaborn import heatmap
import pdb
import os
x0       = (1,1) #Initial conditions. 
tau      = 1000  #Lifespan
nbsteps  = 20    #Number of selective episodes in an organism’s lifetime
w2       = 10    #Inverse selection strength (constant)

V0       = (0.5,0.01)  #Genetic variance (again for l, m)
Vr       = [0.3,0.005] #This is the variability during reproduction, in the infinitesimal model

#Autocorrelation of the environment, careful it is the parameter in the equivalent Ornstein-Uhlenbeck process !
rho      = 1
sig2_e   = 20 #Variance in the environment


Tmax     = 5000
def GenerateEnvironment(r=rho, s2=sig2_e, n = nbsteps*Tmax, time=tau*Tmax):
    """This function, for a given autocorrelation r and environmental standard deviation s, generates an AR[1] sequence of length n
    Keep in mind that a discretized (in steps of size dt) O-U process with parameters (r,s2) is an AR[1] process with parameters (exp(-rho dt), sqrt(s2(1-exp(-2rho dt))))"""
    x= ArmaProcess([1,-np.exp(-r*time/n)]).generate_sample(n,scale=np.sqrt(s2*(1-np.exp(-2 * r *time/n))))
    return x