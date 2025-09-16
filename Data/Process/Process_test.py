import matplotlib.pyplot as plt
import numpy as np
import roughpy as rp
from Events import *
from Heston import *
from Signature import Random_signature as rs

n_samples = 1
n_steps = 10000
T = 1
rate = 1
dim_min_max = (1,1)
rebound = False
rebound_rate = 1

thet = 0.02*np.ones([n_steps, n_samples])
sigma = 0.01*np.ones([n_steps, n_samples])
mu = 0.01*np.ones([n_steps, n_samples])
S0 = 0.03*np.ones([1, n_samples])


loc = Event_location(n_samples, n_steps, T, rate, dim_min_max, rebound=True, rebound_rate=rebound_rate)

path = CIR(thet, sigma, mu, T, n_samples, n_steps, S0)
val = np.ones(len(loc[2]))*sigma[0,0]*4

sigma_shift = Parameter_shift(sigma, loc, val)
path_shift = CIR(thet, sigma_shift, mu, T, n_samples, n_steps, S0)
path_jump = Jump_rebound(path_shift, loc, 0, 0.001)


plt.plot(path, label="Path")
plt.plot(path_shift, label="Path_shift")
plt.plot(path_jump, label="Path_jump")
plt.title("Paths")
plt.legend()
plt.show()

print(loc)




"""
out = rs.Random_signature(path, 100)
out_shift = rs.Random_signature(path_shift, 100)
out_jump = rs.Random_signature(path_jump, 100)

plt.plot(out[:,-3], label="Path")
plt.plot(out_shift[:,-3], label="Path shift")
plt.plot(out_jump[:,-3], label="Path jump")
plt.title("Random Signature")
plt.legend()
plt.show()
"""



"""
times = np.linspace(0.1,1, 1000)
context = rp.get_context(width=1000, depth=3, coeffs=rp.DPReal)
stream = rp.LieIncrementStream.from_increments(path, indices=times, ctx=context)
interval = rp.RealInterval(0, T)
sig = stream.log_signature(interval)
print(f"{sig=!s}")


context = rp.get_context(width=1000, depth=3, coeffs=rp.DPReal)
stream = rp.LieIncrementStream.from_increments(path_shift, indices=times, ctx=context)
interval = rp.RealInterval(0, T)
sig = stream.log_signature(interval)
print(f"{sig=!s}")
"""