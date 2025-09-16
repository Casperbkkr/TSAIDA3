import numpy as np
import matplotlib.pyplot as plt
import roughpy as rp


from TS_AIDA import Window as Wd
from TS_AIDA import AIDA as AIDA
from TS_AIDA import Path_signature as Ps


def Random_subsampler(paths, N, parameters, local=False, window_corrected=False, dim_corrected=False, seed=None, c=None):
    rng = np.random.default_rng()

    length_min_max = (parameters[0], parameters[1])
    dim_min_max = (parameters[2], parameters[3])
    n_samples_min_max = (parameters[4], parameters[5])
    w_ratio = parameters[6]

    sample_n = Sample_n(N, n_samples_min_max, rng)

    if window_corrected == False:
        sample_lengths = Sample_length(N, length_min_max, rng)
    else:
        sample_lengths = Sample_length_corr(N, length_min_max, rng)


    if local == False:
        n_windows = paths.shape[0] - sample_lengths
        indices = [np.arange(0, n_windows[i]) for i in range(N)]

        sample_start = [rng.choice(indices[i], size=sample_n[i], replace=False) for i in range(N)]
        sample_end = [sample_start[i] + sample_lengths[i] for i in range(N)]

    else:
        if c != None:
            print("central point fixed")
            central_points = np.ones(shape=[N])*c
            starts = np.maximum(central_points - w_ratio * sample_lengths, 0)
            ends = np.minimum(central_points + w_ratio * sample_lengths, paths.shape[0] - sample_lengths)
        else:
            central_points = rng.integers(0, paths.shape[0], size=N)
            #starts = np.maximum(central_points - w_ratio * sample_lengths, 0)
            starts = central_points - w_ratio * sample_lengths
            #er = np.where(starts + 2*w_ratio * sample_lengths < paths.shape[0]-2*w_ratio * sample_lengths, starts, paths.shape[0]-2*w_ratio * sample_lengths)
            ends = starts + 2*w_ratio * sample_lengths


        sample_start=[]
        sample_end=[]
        for i in range(N):
            indices = np.arange(starts[i], ends[i])
            sample_starts2 = rng.choice(indices, size=sample_n[i], replace=False)
            diffe = sample_starts2 + sample_lengths[i] - paths.shape[0]
            sample_starts2 = np.where(sample_starts2 + sample_lengths[i] >= paths.shape[0], sample_starts2 - 2*diffe-1, sample_starts2)
            sample_starts2 = np.where(sample_starts2 + sample_lengths[i] < 0, np.abs(sample_starts2 + sample_lengths[i]), sample_starts2)
            sample_start.append(sample_starts2)
            sample_end.append(sample_starts2 + sample_lengths[i])

    if dim_corrected == False:
        if paths.shape[1] > 1:
            sample_d = Sample_dim(N, dim_min_max, paths.shape[1], rng)
        else:
            sample_d = Sample_dim(N, dim_min_max, 1, rng)
    else:
        if paths.shape[1] > 1:
            sample_d = Sample_dim(N, dim_min_max, paths.shape[1], rng)
        else:
            sample_d = Sample_dim_corr(N, dim_min_max, 1, rng)


    return (sample_lengths, sample_start, sample_end, sample_d)




def Sample_n(N, n_samples_min_max, rng):
    return rng.integers(low=n_samples_min_max[0], high=n_samples_min_max[1], size=N)

def Length_corr(N, length_min_max, rng):
    a = np.arange(length_min_max[0], length_min_max[1])

    p0 = 1/(1+a[0]*np.sum(1/a[1:]))
    pn = p0*(a[0]/a)
    return pn

def Sample_dim(N, dim_min_max, D, rng):

    if dim_min_max[1] == 1:
        out = np.zeros(N)
    else:
        n_dims = rng.integers(low=dim_min_max[0], high=dim_min_max[1], size=N)
        out = [rng.choice(D, size=n_dims[i], replace=False) for i in range(len(n_dims))]

    return out


def Sample_dim_corr(N, dim_min_max, D, rng):

    pn = prob_corr(dim_min_max[0], dim_min_max[1])

    if dim_min_max[1] == 1:
        out = np.zeros(N)
    else:
        n_dims = rng.choice(np.arange(dim_min_max[0], dim_min_max[1]), p=pn ,size=N)
        out = [rng.choice(D, size=n_dims[i], replace=False) for i in range(len(n_dims))]

    return out

def Sample_length(N, length_min_max, rng):
    return rng.integers(low=length_min_max[0], high=length_min_max[1], size=N)

def Sample_length_corr(N, length_min_max, rng):
    pn = prob_corr(length_min_max[0], length_min_max[1])
    return rng.choice(np.arange(length_min_max[0], length_min_max[1]), p=pn ,size=N)

def prob_corr(min, max):
    a = np.arange(min, max)

    p0 = 1 / (1 + a[0] * np.sum(1 / a[1:]))
    pn = p0 * (a[0] / a)
    return pn