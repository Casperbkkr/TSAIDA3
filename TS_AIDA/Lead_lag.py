import numpy as np

def lead_lag(X):
    lag = np.empty((X.size + X.size,), dtype=X.dtype)
    lag[0::2] = X
    lag[1::2] = X
    lead = np.insert(lag,-1, lag[-1])
    lead = lead[1:]
    out = np.concatenate([lead[:,np.newaxis], lag[:,np.newaxis]], axis=1)

    return out


