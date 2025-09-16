import pathlib


import matplotlib.pyplot as plt
import numpy as np
import roughpy as rp
import seaborn as sns
sns.set_theme()
import time

from Data.Process import Wiener as wien
from Data.Process import GBM as GBM
K=3
n_experiments = 100
n_samples = 10
n_steps = 3000
T = 1
rate = 4
dim_min_max = (1,3)
rebound = False
rebound_rate = 1
rng =  np.random.default_rng()
the = 0.01
thet = the*np.ones([n_steps, n_samples])
sigm = 0.1
Sigma = sigm*np.ones([n_steps, n_samples])
#Sigma = rng.uniform(0.01, 0.015, size=[n_experiments, n_samples])
muuu = 0.1
mu = muuu*np.ones([n_steps, n_samples])
mu[0:500,:] =-0.3
d=10

s0 = rng.uniform(1, 2, size=[n_experiments, n_samples])




data_set_standard = np.zeros(shape=[n_experiments, n_steps, n_samples])
data_set_jump = np.zeros(shape=[n_experiments, n_steps, n_samples])
anom_indices = np.zeros(shape=[n_experiments, n_steps])



for i in range(n_experiments):
    S0 = s0[i,:]
    #mu = Mu[i,:]
    sigma = Sigma[i,:]


    path = GBM.GBM(S0, mu, sigma, T, n_samples, n_steps)[0][:-1]
    data_set_standard[i,:,:] = path
    data_set_jump[i, :, :] = path


    #event_info = ev.Event_location(n_samples, n_steps, T, rate, (1, 3), rebound=True, rebound_rate=rebound_rate)

   # A = np.zeros(shape=[n_steps])
    #for j in range(event_info[0].shape[0]):
     #   A[event_info[0][j]:event_info[1][j]] = 1

    #anom_indices[i, :] = A
    path_j = path#[:, 0][:,np.newaxis]
    pos_neg = rng.choice([1,0.98/1.02], p=[0.5, 0.5], size=d)
    path_jump = pos_neg*1.02*path[int(n_steps/2):int(n_steps/2+20), :d]
    data_set_jump[i,int(n_steps/2):int(n_steps/2+20),:d] = path_jump



total_out = np.zeros(shape=[n_experiments, n_steps, n_samples,2])
total_out[:,:,:,0] =data_set_standard
total_out[:,:,:,1] = data_set_jump

np.save( "/Users/casperbakker/PycharmProjects/TSAIDA2/Tests/Data_input/Jump/data", total_out)
np.save( "/Users/casperbakker/PycharmProjects/TSAIDA2/Tests/Data_input/Jump/anom", anom_indices)



