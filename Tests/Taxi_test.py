import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme()

from TS_AIDA import Subsampling as ss
from TS_AIDA import Random_subsampling as ss2

from TS_AIDA import Classifier as cl



# Local search
local = True
# Local search width ratio
delta = 5
# Corrected sampling
window_correction = True
dim_correction = True
# number of random subsamples
N=1000
# Path signature truncation order
K=3
# Fourier thresholding percentage of discarded terms
R_dft=0.6
# Exclusion ratio
r_exc=1

# Min and max window length
l_min = 24
l_max = 4*7*24
# Min and max subspace sample l=size
d_min = 1
d_max = 2
# Number of subsamples
m_min = 20
m_max = 30


parameters = (l_min, l_max, d_min, d_max, m_min, m_max, delta)

# Load data and labeld anomalies
data = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Tests/Data_input/NYC/NYC_data.npy")[:,np.newaxis]
anom = np.load("/Users/casperbakker/PycharmProjects/PythonProject/Tests/Data_input/NYC/NYC_anomaly.npy")

# Create random subsamples
sample_info = ss2.Random_subsampler(data, N=N, parameters=parameters, local=local, window_corrected=window_correction, dim_corrected=dim_correction)
# Calculate score profiles
dis_local_score_profile, _, dis_global_score_profile = ss.Score_aggregator(data, sample_info, K, 1, r=delta)
dft_local_score_profile, _, dft_global_score_profile = ss.Score_aggregator(data, sample_info, K, 1, r=delta, fourier=True, thresh=R_dft)
sig_local_score_profile, _, sig_global_score_profile = ss.Score_aggregator(data, sample_info, K, 1, r=delta, sig=True)

# Calculate performance and optimal threshold value alpha.
alpha_dis, score_dis = cl.Max_perf1(dis_local_score_profile[50:-50], anom[50:-50])
alpha_dft, score_dft = cl.Max_perf1(dft_local_score_profile[50:-50], anom[50:-50])
alpha_sig, score_sig = cl.Max_perf1(sig_local_score_profile[50:-50], anom[50:-50])

plt.plot(dis_local_score_profile[50:-50], label="$F_{0.5}^{event} =$"+str(score_dis[-1]))
plt.title("TS-AIDA-Dis")
plt.xlabel("Time")
plt.ylabel("Anomaly score")
plt.axhline(y=alpha_dis[-1], color='r', linestyle='-', label="$\\alpha$")
plt.legend()
plt.show()

plt.plot(dft_local_score_profile[50:-50], label="$F_{0.5}^{event} =$"+str(score_dft[-1]))
plt.title("TS-AIDA-DFT")
plt.xlabel("Time")
plt.ylabel("Anomaly score")
plt.axhline(y=alpha_dft[-1], color='r', linestyle='-', label="$\\alpha$")
plt.legend()
plt.show()

plt.plot(sig_local_score_profile[50:-50], label="$F_{0.5}^{event} =$"+str(score_sig[-1]))
plt.title("TS-AIDA-Sig")
plt.xlabel("Time")
plt.ylabel("Anomaly score")
plt.axhline(y=alpha_dft[-1], color='r', linestyle='-', label="$\\alpha$")
plt.legend()
plt.show()

