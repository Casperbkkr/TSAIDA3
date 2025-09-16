import numpy as np

def GBM(S0, r, sigma, T, n_samples, n_steps):

    dt = float(T) / n_steps
    rng = np.random.default_rng(seed=1)
    rand = rng.normal(loc=0, scale=1, size=[n_steps, n_samples])

    A = (r-0.5*sigma**2)*dt + sigma * np.sqrt(dt)*rand

    B = np.exp(A)

    C = S0*np.ones(shape=(1, n_samples))

    D = np.concatenate((C, B), axis=0)

    return np.cumprod(D, axis=0), D

def GBM_Jumps(paths, T, rate, mean, var):
    n_steps, n_samples = paths.shape

    dt = float(T) / n_steps
    rate_dt = rate*dt
    rng = np.random.default_rng()

    jump_index = rng.poisson(lam=rate_dt, size=paths.shape)
    jump_size = rng.normal(loc=mean, scale=np.sqrt(var), size=paths.shape)
    jumps = np.multiply(jump_index, jump_size)


    jump_paths = np.zeros_like(paths)
    jump_paths[0, :] = paths[0, :]

    for step in range(1, n_steps):
        jump_paths[step, :] = jump_paths[step-1,:]*paths[step,:] + paths[step-1,:]*jumps[step-1,:]



    x, y  = np.where(jump_index == 1)
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    x = x-1
    z = np.take_along_axis(jump_paths, x, axis=0)

    q = np.take_along_axis(z, y, axis=1)
    jump_loc = np.concatenate((x, q), axis=1)

    return jump_loc, jump_paths

def GBM_jumps_change(path, T, mean, var, event_start, event_end):
    n_steps, n_samples = path.shape

    dt = float(T) / n_steps

    rng = np.random.default_rng()


    jump_size = rng.normal(loc=mean, scale=np.sqrt(var), size=path.shape)
    jumps = np.multiply(event_start, jump_size)
    jumps_end = np.multiply(event_end, 1/jump_size)
    jump_paths = np.zeros_like(path)
    jump_paths[0, :] = path[0, :]

    normal = True
    for step in range(1, n_steps):
        if normal == True:
            jump_paths[step, :] = jump_paths[step - 1, :] * path[step, :] + path[step - 1, :] * jumps[step - 1, :]
            if jump_index[step,:] == 1:
                jump_paths[step, :] = jump_paths[step - 1, :] * path[step, :] + path[step - 1, :] * jumps[step - 1, :]
                normal = False
        else:
            jump_paths[step, :] = jump_paths[step - 1, :] * path_var[step, :] + path_var[step - 1, :] * jumps[step - 1, :]
            if jump_index[step,:] == 1:
                normal = True

    x, y = np.where(jump_index == 1)
    x, y = x[:, np.newaxis], y[:, np.newaxis]
    x = x - 1
    z = np.take_along_axis(jump_paths, x, axis=0)

    q = np.take_along_axis(z, y, axis=1)
    jump_loc = np.concatenate((x, q), axis=1)

    return jump_loc, jump_paths