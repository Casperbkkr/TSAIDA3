import pathlib
import time

import numpy as np


import seaborn as sns
from matplotlib import pyplot as plt

from Data.Process import Wiener as Wiener
from Data.Process import Heston as He
sns.set_theme()

T = 1
dim = 10
n_steps = 2000
K=4
n_experiments=100



sin_out = np.zeros(shape=[n_experiments, n_steps, dim, 2])
anom_out = np.zeros(shape=[n_experiments, n_steps])
start_anom = int(n_steps / 2)
end_anom = int(start_anom + n_steps / 50)

rng = np.random.default_rng()

for j in range(n_experiments):

    x1 = np.linspace(0, T, n_steps)
    x2 = x1[:,np.newaxis]
    x3 = np.repeat(x2, dim, axis=1)

    trans_para = 20
    trans = rng.uniform(0,trans_para, size=dim)
    trans = trans[np.newaxis, :]
    trans = np.repeat(trans, n_steps, axis=0)



    c_range=[]

    rho =  0.7*np.ones(shape=(n_steps, 2*dim))


    th_1 = rng.uniform(0.29,0.30, dim)
    th_2 = th_1 + 0.3
    S0 = rng.uniform(0.29,0.32, dim)

    theta = th_1 * np.ones([n_steps, dim])
    theta_anom = th_1 * np.ones([n_steps, dim])
    theta_anom[1000:1040] = th_2
    sigma = 10.61 * np.ones([n_steps, dim])




    for i in range(int(n_steps/100)):
        c_range.append([i * 100, (i + 1) * 100])


    W, W_hes= Wiener.Correlated_Wiener(rho, T, n_steps, dim, c_range=c_range)

    #plt.plot(W)
    #plt.show()

    kappa = 16.21 * np.ones([n_steps, dim])

    xi = 0.061 * np.ones([n_steps, dim])

    cir = He.CIR(kappa, xi, theta, T, dim, n_steps, S0, W)
    cir2 = He.CIR(kappa, xi, theta_anom, T, dim, n_steps, S0, W)

    anom = np.zeros(shape=n_steps)
    anom_data = np.zeros(shape=(n_steps, dim))
    anom[start_anom:end_anom] = 1

    sin_normal = cir #+ trans
    sin_anom = cir2 #+ trans # noise3
    S0 = rng.uniform(1,2,size=dim)
    heston = He.Heston(cir, 0.31, T, n_steps, S0, W_hes)
    heston_anom = He.Heston(cir2, 0.31, T, n_steps, S0, W_hes)

    sin_out[j,:,:,0] =heston
    sin_out[j,:,:,1] =heston_anom
    anom_out[j,:] = anom



np.save("/Users/casperbakker/PycharmProjects/TSAIDA2/Tests/Data_input/Heston/data", sin_out)
np.save("/Users/casperbakker/PycharmProjects/TSAIDA2/Tests/Data_input/Heston/anom", anom_out)
