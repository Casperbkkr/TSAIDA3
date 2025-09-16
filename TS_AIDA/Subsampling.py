import gc

import numpy as np
import matplotlib.pyplot as plt
import roughpy as rp


from TS_AIDA import Window as Wd
from TS_AIDA import AIDA as AIDA
from TS_AIDA import Path_signature as Ps

def sampler(paths, sample_info, i):
    paths = paths[:, sample_info[3][i]]

    if len(paths.shape) > 2:
        reps = paths.shape[2]
    else:
        reps = 1
    # hier gemaakt zodat meer dims in een keer.
    paths = paths[:, np.newaxis, :]
    windows = Wd.sliding_window(paths, sample_info[0][i], 1).transpose(0, 2, 1)
    A = np.repeat(sample_info[1][i][:, np.newaxis], paths.shape[2], axis=1)[:, np.newaxis, :]
    A = A.astype(int)
    subsample = np.take_along_axis(windows, A, axis=0)

    return subsample


def Exclusion_zone(sample_info, i, r):

    #r = 1-ratio
    starts = sample_info[1][i]
    starts = np.repeat(starts[np.newaxis, :], starts.shape[0], axis=0)

    ends = sample_info[2][i]
    ends = np.repeat(ends[np.newaxis, :], starts.shape[0], axis=0)

    overlap = (ends - starts.transpose())
    overlap = np.where(overlap > 0, overlap, 0)
    overlap2 = (starts - ends.transpose())
    overlap2= np.where(overlap2 < 0, np.abs(overlap2), 0)

    overlap = np.where(overlap < sample_info[0][i], overlap, 0)
    overlap2 = np.where(overlap2 < sample_info[0][i], overlap2, 0)

    total_overlap = overlap + overlap2
    R = total_overlap/sample_info[0][i]
    R = np.where(np.abs(R) > r, 0, 1)
    R = R*np.arange(0,starts.shape[1])#[i for i in range(starts.shape[1])]
    return R

def Score_aggregator(paths, sample_info, K, T, normalize=False, sig=False, fourier=False, r=0.6, thresh = 0.5):
    N = sample_info[0].shape[0]
    output = np.zeros(shape=[paths.shape[0],2])
    output_global = np.zeros(shape=[paths.shape[0]])
    for i in range(N):
        #if i %50 == 0: print(i)
        score_sub = np.zeros(shape=[paths.shape[0], 2])

        s1 = sampler(paths, sample_info, i)
        excl = Exclusion_zone(sample_info, i, r)
        if normalize is True:
            s2 = (s1 - np.mean(s1, axis=0)) / s1.std(axis=0)
        if normalize is False:
            s2 = s1

        if fourier == True:

            s4 = AIDA.DFT(s2, thresh)[0]
        else:
            s4 = s2

        if sig == True:
            K2 =  K#max(2, K-len(sample_info[3][i]))
            interval = rp.RealInterval(0, sample_info[0][i])
            indices = np.linspace(0.1, T, sample_info[0][i])
            context = rp.get_context(width=s2.shape[2], depth=K2, coeffs=rp.DPReal)

            s4 = Ps.sig_rp2(s2, K2, interval, indices, context)
            #interval, indices, context = None, None, None

        if sig == False and fourier == False:
            s4 = s2



        contrast = np.zeros(shape=[s4.shape[0], 1])

        for j in range(s4.shape[0]):
            include = excl[j]
            include2 =  include[include>0]
            s5 = np.take(s4, include2, axis=0)
            score_mean, score_var, cont = AIDA.Score(s4[j, :], s5)
            contrast[j] = cont
            a = sample_info[1][i][j]
            b = sample_info[2][i][j]

            score_sub[a:b, 0] -= score_var
            score_sub[a:b, 1] += 1

        #score_sub[:, 1] = np.where(score_sub[:, 1] == 0, 1, score_sub[:, 1])
            # score_sub[:, 0] = (score_sub[:, 0] - score_sub[:, 0].mean())/ score_sub[:, 0].std()
        output[:, 0] += score_sub[:, 0]
        output[:, 1] += score_sub[:, 1]

        W_len = sample_info[0][i]
        corrector = W_len*np.ones_like(score_sub[:, 0])
        corrector[:W_len-1]= np.arange(1,W_len)
        G = np.flip(np.arange(1,W_len+1))
        corrector[-(W_len):] = G

        output_global[:] += score_sub[:, 0]/corrector
        # /score_sub[:,1]
            # score_sub[:, 0] = np.nan_to_num(score_sub[:, 0])

    output[:, 1] = np.where(output[:, 1] == 0, 1, output[:, 1])
    output_avg = output[:, 0] / output[:, 1]
    output_global = (output_global - output_global.mean()) / output_global.std()
    output_avg = (output_avg - output_avg.mean()) / output_avg.std()

    return output_avg, contrast, output_global

def Score_aggregator_PRISM(paths, sample_info, K, T, normalize=False, sig=False, fourier=False, r=0.6):
    N = sample_info[0].shape[0]
    output = np.zeros(shape=[paths.shape[0],2, paths.shape[1]])
    output_global = np.zeros(shape=[paths.shape[0], paths.shape[1]])
    for i in range(N):
        #if i %50 == 0: print(i)
        score_sub = np.zeros(shape=[paths.shape[0], 2, (sample_info[3][i]).shape[0]])

        s1 = sampler(paths, sample_info, i)
        excl = Exclusion_zone(sample_info, i, r)
        if normalize is True:
            s2 = (s1 - np.mean(s1, axis=0)) / s1.std(axis=0)
        if normalize is False:
            s2 = s1

        if fourier == True:
            thresh = 100
            s3 = AIDA.DFT(s2, thresh)
        else:
            s3 = s2

        if sig == True:
            K2 =  K#max(2, K-len(sample_info[3][i]))
            interval = rp.RealInterval(0, sample_info[0][i])
            indices = np.linspace(0.1, T, sample_info[0][i])
            context = rp.get_context(width=s3.shape[2], depth=K2, coeffs=rp.DPReal)

            s4 = Ps.sig_rp2(s3, K2, interval, indices, context)
            #interval, indices, context = None, None, None

        if sig == False:
            s4 = s3



        contrast = np.zeros(shape=[s4.shape[0], 1])

        for j in range(s4.shape[0]):
            include = excl[j]
            include2 =  include[include>0]
            s5 = np.take(s4, include2, axis=0)
            score_mean, score_var, cont = AIDA.Score(s4[j, :], s5)
            contrast[j] = cont
            a = sample_info[1][i][j]
            b = sample_info[2][i][j]

            score_sub[a:b, 0] -= score_var
            score_sub[a:b, 1] += 1

        #score_sub[:, 1] = np.where(score_sub[:, 1] == 0, 1, score_sub[:, 1])
            # score_sub[:, 0] = (score_sub[:, 0] - score_sub[:, 0].mean())/ score_sub[:, 0].std()
        a=sample_info[3][i]
        b = output[:, 0, sample_info[3][i]]
        c = score_sub[:, 0]
        output[:, 0, sample_info[3][i]] += score_sub[:, 0]
        output[:, 1, sample_info[3][i]] += score_sub[:, 1]

        W_len = sample_info[0][i]
        corrector = W_len*np.ones_like(score_sub[:, 0,:])
        arran = np.arange(1,W_len)[:, np.newaxis]
        corrector[:W_len-1]= np.repeat(arran, score_sub.shape[2],axis=1)
        G = np.flip(arran)
        G = np.repeat(G, score_sub.shape[2], axis=1)
        corrector[-(W_len)+1:] = G
        gdas = score_sub[:, 0, :]/corrector
        output_global[:, sample_info[3][i]] += score_sub[:, 0, :]/corrector
        # /score_sub[:,1]
            # score_sub[:, 0] = np.nan_to_num(score_sub[:, 0])

    output[:, 1] = np.where(output[:, 1] == 0, 1, output[:, 1])
    output_avg = output[:, 0] / output[:, 1]
    output_global = (output_global - output_global.mean()) / output_global.std()
    output_avg = (output_avg - output_avg.mean()) / output_avg.std()

    return output_avg, contrast, output_global

