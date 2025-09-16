import numpy as np
def sliding_window(path, size, step):
    windows = np.lib.stride_tricks.sliding_window_view(path, size, axis=0)
    windows = windows[1::step]
    return windows [:,0,:]