import pathlib
import time

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


epoch_time = int(time.time())
sns.set_theme()

T = 200
dim = 10
steps = 2000
dt = T/steps

n_experiments = 100
n_events = 200

n_anomalous_events = 1
sin_out = np.zeros(shape=[n_experiments, steps, dim, 2])
anom_out = np.zeros(shape=[n_experiments, steps])

rng = np.random.default_rng()

for i in range(n_experiments):

    x1 = np.linspace(0, T, steps)
    x2 = x1[:,np.newaxis]
    x3 = np.repeat(x2, dim, axis=1)

    trans_para = 20
    trans = rng.uniform(0,trans_para, size=dim)
    trans = trans[np.newaxis, :]
    trans = np.repeat(trans, steps, axis=0)

    amp_low = 1
    amp_high = 6
    amp = rng.uniform(amp_low, amp_high, size=dim)
    amp = amp[np.newaxis, :]
    amp = np.repeat(amp, steps, axis=0)


    noise_scale = 0.1
    cauchy_scale = 0#.001
    noise1 = rng.normal(loc=0.0, scale=noise_scale, size=x3.shape)



    start_anom = int(steps/2)
    end_anom = int(start_anom+steps/50)
    anom = np.zeros(shape=steps)
    anom[start_anom:end_anom] = 1

    ad = np.arange(0, steps-30, dtype=int)
    starts = rng.choice(ad, n_events)
    starts = starts.tolist()

    period_low = 3
    period_high = 4

    G = np.zeros_like(x3)
    anom_starts = []
    used = np.zeros(shape=[steps])
    for k in range(n_events):

        event = np.zeros_like(G)

        start = starts[k]

        amp_low = 1
        amp_high = 2
        amp = rng.uniform(amp_low, amp_high, size=1)
        amp = np.repeat(amp, dim, axis=0)
        amp = amp[np.newaxis, :]
        amp = np.repeat(amp, steps, axis=0)

        B = rng.uniform(period_low, period_high, size=[1, dim])
        #B[:,:] = B[0, 0]
        period = 2*np.pi/B
        sines = amp*-1*np.sin(x3 * B)
        f = period / dt
        r1 = f.astype(np.int32)
        max_length = np.max(r1)
        f = used[start:start + max_length]

        if np.sum(used[start:start + max_length]) > 0:
            continue
        else:
            for d in range(0, dim):
                length = r1[:,d][0]
                sine_part = sines[:length, d]
                event[start:start+length, d] = sine_part
            used[start:start + max_length] = 1

        G += event
        anom_starts.append(start)
    G2 = np.zeros_like(G)

    for k in range(n_anomalous_events):
        start_ind = np.where(np.array(anom_starts) > 750)[0]
        start_ind2 = np.where(np.array(anom_starts) <1250)[0]
        start_ind = list((set(start_ind) & set(start_ind2)))[0]
        start = anom_starts[start_ind]
        count =1

        amp = rng.uniform(amp_low, amp_high, size=dim)
        amp = amp[np.newaxis, :]
        amp = np.repeat(amp, steps, axis=0)

        #B = rng.uniform(period_low, period_high, size=[1, dim])
        #B[:,:] = 5# B[0,0]
        period = 2*np.pi/B
        cosines = amp*np.sin(x3 * B)
        f = period / dt
        r = f.astype(np.int32)


        length = f.astype(np.int32)
        cosine_part = cosines[:np.max(length)+1, :]
        a=np.max(length)
        G2[start:start+np.max(length)+1,:] = cosine_part
        G2[:start,:] = G[:start,:]
        G2[start+np.max(length)+1:, :] = G[start+np.max(length)+1:, :]
        A = np.sum(G[start:start+np.max(length)+1] - G2[start:start+np.max(length)+1])
        anom_out[i,start:start + np.max(length)+1] = 1


    #period[start_anom:end_anom, :5] = period[start_anom:end_anom, :5]/5
    sin_out[i,:,:,0] = noise1  + G + trans
    sin_out[i,:,:,1] = noise1 + G2 +trans




np.save("/Users/casperbakker/PycharmProjects/TSAIDA2/Tests/Data_input/Order/data", sin_out)
np.save("/Users/casperbakker/PycharmProjects/TSAIDA2/Tests/Data_input/Order/anom", anom_out)