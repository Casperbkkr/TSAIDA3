import numpy as np
from Data.Process import Wiener as wien

def CIR(theta, sigma, mu, T, n_samples, n_steps, S0, Wt=np.array([0.0])):
    if Wt.shape[0] == 1:
        Wt = wien.Wiener(T, n_samples, n_steps).transpose()
    dt = float(T) / n_steps

    #rng = np.random.default_rng()
    #rand = rng.normal(loc=0, scale=1, size=[n_steps, n_samples])
    #Wt = np.sqrt(dt) * rand

    A = 1-theta*dt
    B = theta*mu*dt
    C = sigma*Wt

    paths = np.zeros(shape=(n_steps, n_samples))
    paths[0,:] = S0
    for i in range(1, n_steps):
        paths[i,:] = paths[i-1,:]*A[i-1,:] + B[i-1,:] + C[i-1,:]*np.sqrt(paths[i-1,:])

    return paths

def CIR_shift(theta, sigma, mu, T, n_samples, n_steps, S0):

    dt = float(T) / n_steps
    rng = np.random.default_rng()
    rand = rng.normal(loc=0, scale=1, size=[n_steps, n_samples])
    Wt = np.sqrt(dt) * rand

    A = 1-theta*dt
    B = theta*mu*dt
    C = sigma*Wt

    paths = np.zeros(shape=(n_steps, n_samples))
    paths[0,:] = S0
    for i in range(1, n_steps):
        paths[i,:] = paths[i-1,:]*A[i-1,:] + B[i-1,:] + C[i-1,:]*np.sqrt(paths[i-1,:])

    return paths

def Heston(v, mean, T, n_steps, S0, W_hes):
    n_samples = v.shape[1]
    v = np.maximum(v, 0)
    dt = float(T) / n_steps


    Wt = W_hes

    A = (mean-0.5*v)*dt
    B = np.sqrt(v*dt)*Wt/dt

    paths = np.zeros(shape=(n_steps, n_samples))
    paths[0, :] = S0

    for i in range(1, n_steps):
        paths[i, :] = paths[i - 1, :] + A[i - 1, :] + B[i - 1, :]

    return paths