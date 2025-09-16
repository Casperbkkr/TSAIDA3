import numpy as np
import roughpy as rp
from TS_AIDA import Random_subsampling as ss2
from TS_AIDA import Subsampling as ss
import matplotlib.pyplot as plt
from TS_AIDA import Window as Wd
from TS_AIDA import AIDA as AIDA
from TS_AIDA import Path_signature as Ps

def TS_AIDA(data, parameters, N, K, T, R, local=False, window_corrected=False, dim_corrected=False):
    plt.plot(data)
    plt.title("Example")
    plt.show()
    print("Data_output loaded")

    sample_info = ss2.Random_subsampler(data, N=N, parameters=parameters, local=local, window_corrected=window_corrected,
                                              dim_corrected=dim_corrected)

    sig_score, _, sig_score_global = ss.Score_aggregator(data, sample_info, K, T, sig=True)
    dft_score, _, dft_score_global = ss.Score_aggregator(data, sample_info, K, T, fourier=True, R=R)
    dis_corr, _, dis_corr_N = ss.Score_aggregator(data, sample_info, K, T, sig=False)


    return sig_score, sig_score_global, dft_score, dft_score_global, dis_corr, dis_corr_global

def Subsample_examiner(data, startx, endx, dims, n_samples, w_ratio=5, K=4):
    rng = np.random.default_rng(42)
    length = endx - startx

    # Global
    n_windows = data.shape[0] - length
    indices = np.arange(0, n_windows)
    local_sample_start = rng.choice(indices, size=n_samples, replace=False)

    windows = Wd.sliding_window(data, length, 1)

    global_sample = np.zeros(shape=(n_samples, length, len(dims)))


    global_sample[0,:,:] = data[startx:endx, dims]
    for i in range(1, n_samples):
        start = local_sample_start[i]
        end = start + length
        global_sample[i, :, :] = data[start:end, dims]


    search_area_center = startx + length/2
    area_left = int(search_area_center - w_ratio*length)
    area_right = int(search_area_center + w_ratio*length)
    search_area = data[area_left:area_right, dims]

    area_length = area_right - area_left
    n_windows = area_length - length

    indices = np.arange(0, n_windows)
    local_sample_start = area_left + rng.choice(indices, size=n_samples, replace=False)
    local_sample = np.zeros(shape=(n_samples, length, len(dims)))

    local_sample[0, :, :] = data[startx:endx, dims]

    for i in range(1, n_samples):
        start = local_sample_start[i]
        end = start + length
        local_sample[i, :, :] = data[start:end, dims]


    interval = rp.RealInterval(0, length)
    indices = np.linspace(0.1, 1, length)
    context = rp.get_context(width=len(dims), depth=K, coeffs=rp.DPReal)

    global_sample_sig = Ps.sig_rp2(global_sample, K, interval, indices, context)
    local_sample_sig = Ps.sig_rp2(local_sample, K, interval, indices, context)

    plt.plot(global_sample_sig[0])
    plt.plot(local_sample_sig[1])
    plt.plot(global_sample_sig[1])
    plt.show()


    dp_global_sig = AIDA.DistanceProfile(global_sample_sig[0], global_sample_sig)
    S1 = AIDA.Isolation(dp_global_sig)
    dp_local_sig = AIDA.DistanceProfile(local_sample_sig[0], local_sample_sig)
    S2 = AIDA.Isolation(dp_local_sig)

    dp_global = AIDA.DistanceProfile(global_sample[0], global_sample)
    S3 = AIDA.Isolation(dp_global)
    dp_local = AIDA.DistanceProfile(local_sample[0], local_sample)
    S4 = AIDA.Isolation(dp_local)

    y = np.ones(n_samples)

    plt.scatter(dp_global_sig/dp_global_sig.max(), 1 * y, label="Signature" + str(S1[1]))
    plt.scatter(dp_global/dp_global.max(), 2 * y, label="Distance"+ str(S3[1]))
    plt.title("Global distance profile")
    plt.legend()
    plt.show()

    plt.scatter(dp_local_sig/dp_local_sig.max(), 1 * y, label="Signature"+ str(S2[1]))
    plt.scatter(dp_local/dp_local.max(), 2 * y, label="Distance"+ str(S4[1]))
    plt.title("Local distance profile")
    plt.legend()
    plt.show()

    return dp_local, dp_global, dp_local_sig, dp_global_sig
    #return dp_local/dp_local.max(), dp_global/dp_global.max(), dp_local_sig/dp_local_sig.max(), dp_global_sig/dp_global_sig.max()
