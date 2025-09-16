import pathlib
import time

import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from Data.Process import Wiener as Wiener
from Data.Process import Heston as He
sns.set_theme()

T = 3600
dim = 10
n_steps = 2000
K=4
n_experiments=100

rate = 0.0008*np.ones(n_steps)
decay_rate = 0.3*np.ones(n_steps)
excite_rate =0.7*np.ones(n_steps)

sin_out = np.zeros(shape=[n_experiments, n_steps, dim, 2])
anom_out = np.zeros(shape=[n_experiments, n_steps])
start_anom = int(n_steps / 2)
end_anom = int(start_anom + n_steps/50)

rng = np.random.default_rng()

for i in range(n_experiments):

    x1 = np.linspace(0, T, n_steps)
    x2 = x1[:,np.newaxis]
    x3 = np.repeat(x2, dim, axis=1)

    trans_para = 20
    trans = rng.uniform(0,trans_para, size=dim)
    trans = trans[np.newaxis, :]
    trans = np.repeat(trans, n_steps, axis=0)




    c_range = [[0, start_anom], [start_anom, end_anom], [end_anom, n_steps-1]]
    #c_range = [[0, 300], [300, 600], [600, -1]]
    rho = np.zeros(shape=(n_steps, dim))
    rhos = [0, 0.9, 0]
    for y in range(len(c_range)):
        ranges = c_range[y]
        rho[ranges[0]:ranges[1], :] = rhos[y]


    noise_scale = 0.1
    cauchy_scale = 0.01

    theta = 0.01 * np.ones([n_steps, dim])
    sigma = 0.001 * np.ones([n_steps, dim])
    mu = 0.3*np.ones([n_steps, dim])
    mu[start_anom:end_anom, :] =0.305
    W = Wiener.Wiener(T, n_steps, dim)
    W_anom = Wiener.Correlated_Wiener2(rho, T, n_steps, dim, c_range=c_range)[0]

    noise = W #+ noise2

    anom = np.zeros(shape=n_steps)
    anom_data = np.zeros(shape=(n_steps, dim))
    anom[start_anom:end_anom] = 1




    sin_out[i,:,:,0] = W
    sin_out[i,:,:,1] = W_anom
    anom_out[i,:] = anom




np.save("/Users/casperbakker/PycharmProjects/TSAIDA2/Tests/Data_input/Correlated_noise/data", sin_out)
np.save("/Users/casperbakker/PycharmProjects/TSAIDA2/Tests/Data_input/Correlated_noise/anom", anom_out)