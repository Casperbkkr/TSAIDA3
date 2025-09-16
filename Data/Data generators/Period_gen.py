import pathlib
import time

import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns



sns.set_theme()
T = 1000
dim = 10
steps = 2000
K=4
n_experiments=100

sin_out = np.zeros(shape=[n_experiments, steps, dim, 2])
anom_out = np.zeros(shape=[n_experiments, steps])

rng = np.random.default_rng(42)

for i in range(n_experiments):
    x1 = np.linspace(0, T, steps)
    x2 = x1[:,np.newaxis]
    x3 = np.repeat(x2, dim, axis=1)

    trans_para = 20
    trans = rng.uniform(0,20, size=dim)
    trans = trans[np.newaxis, :]
    trans = np.repeat(trans, steps, axis=0)

    period_low = 3
    period_high = 6*np.pi
    period = rng.uniform(period_low, period_high, size=dim)
    period = period[np.newaxis, :]
    period = np.repeat(period, steps, axis=0)

    shift_para = period_high
    shift = rng.uniform(0, shift_para, size=dim)
    shift = shift[np.newaxis, :]
    shift = np.repeat(shift, steps, axis=0)

    amp_low = 1
    amp_high = 3
    amp = rng.uniform(amp_low, amp_high, size=dim)
    amp = amp[np.newaxis, :]
    amp = np.repeat(amp, steps, axis=0)



    noise11 = rng.normal(loc=0.0, scale=0.0, size=x3.shape)



    start_anom = int(steps/2)
    end_anom = int(start_anom+steps/50)
    anom = np.zeros(shape=steps)
    anom[start_anom:end_anom] = 1



    sin = amp * np.sin(x3 / period + shift) + trans
    sin_out[i, :, :, 0] = sin
    anom_out[i, :] = anom

    amp[start_anom:end_anom, :] = amp[start_anom:end_anom, :] * rng.choice([2,0.1], size=dim, p=[0.5, 0.5])

    sin = amp * np.sin(x3 / period + shift) + trans #+ noise11
    sin_out[i, :, :, 1] = sin




np.save("/Users/casperbakker/PycharmProjects/TSAIDA2/Tests/Data_input/Period/data.npy", sin_out)
np.save("/Users/casperbakker/PycharmProjects/TSAIDA2/Tests/Data_input/Period/anom.npy", anom_out)


