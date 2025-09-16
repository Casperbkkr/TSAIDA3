import numpy as np

def Event_location(n_samples, n_steps, T, rate, dim_min_max, rebound=False, rebound_rate=5):
    rng = np.random.default_rng()

    dt = float(T) / n_steps
    rate_dt = rate * dt
    rebound_rate_dt  = rebound_rate*dt
    event_location = rng.poisson(lam=rate_dt, size=n_steps)
    event_index = np.where(event_location == 1)[0]
    n_events = np.sum(event_location)
    choice_dims = [i for i in range(dim_min_max[0], dim_min_max[1]+1)]
    n_dims = rng.choice(choice_dims, size=n_events)

    dims = [i for i in range(0, n_samples)]
    #event_dim = [rng.choice(dims, replace=False, size=n_dims[i]) for i in range(n_events)]
    event_dim = [0 for i in range(n_events)]
    if rebound == True:
        ret = rng.poisson(lam=rebound_rate_dt, size=(n_events, n_steps))
        ret[:,-1] = 1
        indices = []

        for i in range(n_events):
            # TODO door de poisson komt er niet altijd een 1 in ret.
            indices_i = np.where(ret[i,:] == 1)
            if indices_i[0][0] > n_steps/50:
                indices.append(int(n_steps/50))
            else:
                indices.append(indices_i[0][0])

        event_length = np.array(indices)

        rebound_index = event_index + event_length[0:event_index.shape[0]]
        rebound_index[np.where(rebound_index >= n_steps)[0]] = n_steps

    out = np.zeros(shape=(n_steps, n_samples))
    out_end = np.zeros(shape=(n_steps, n_samples))

    for i in range(n_events):
        dim = event_dim[i]
        event_start = event_index[i]
        out[event_start, dim] = 1
        if rebound == True:
            event_end = np.min((rebound_index[i], n_steps-1))
            out_end[event_end, dim] = 1

    if rebound == True:
        return event_index, rebound_index, event_dim

    if rebound == False:
        return event_index, 0, event_dim

def Parameter_shift(parameter, event_data, val):
    event_index, rebound_index, event_dim = event_data
    parameter_out = parameter.copy()
    for i in range(len(event_dim)):
        a = np.ones([rebound_index[i] - event_index[i], len(event_dim[i])])*val[i]
        parameter_out[event_index[i]:rebound_index[i], event_dim[i]] = a


    return parameter_out

def Jump_point(path, event_data, mean, var):
    event_index, rebound_index, event_dim = event_data
    n_steps, n_samples = path.shape
    rebound_index = event_index + 1

    rng = np.random.default_rng()

    jump_size = rng.normal(loc=mean, scale=np.sqrt(var), size=path.shape)
    jump = np.zeros(shape=(n_steps, n_samples))
    for i in range(len(event_dim)):
        jump[event_index[i], event_dim[i]] = jump_size[event_index[i], event_dim[i]]
    jumps = jump

    jump_paths = path.copy()
    jump_paths[0, :] = path[0, :]

    in_jump = False


    for jump, rebound in zip(event_index, rebound_index):
            # TODO finish this as alternative
        if in_jump == False:
            jump_paths[jump:rebound, :] = path[jump:rebound, :] + np.repeat(jumps[jump, :], repeats=rebound-jump, axis=0)*path[jump, :]

    return jump_paths

def Jump_rebound(path, event_data, mean, var):
    event_index, rebound_index, event_dim = event_data
    n_steps, n_samples = path.shape


    rng = np.random.default_rng()

    jump_size = rng.normal(loc=mean, scale=np.sqrt(var), size=path.shape)
    jump = np.zeros(shape=(n_steps, n_samples))
    for i in range(len(event_dim)):
        jump[event_index[i], event_dim[i]] = jump_size[event_index[i], event_dim[i]]
    jumps = jump

    jump_paths = path.copy()
    jump_paths[0, :] = path[0, :]

    in_jump = False

    for jump1, rebound in zip(event_index, rebound_index):
        # TODO finish this as alternative
        a = path[jump1:rebound, :]
        c =(jumps[jump1, :]*path[jump1, :])[np.newaxis,:]
        b = np.repeat(c, repeats=rebound - jump1, axis=0)
        jump_paths[jump1:rebound, :] = a + b

    return jump_paths

