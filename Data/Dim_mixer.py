import numpy as np
import matplotlib.pyplot as plt


def Mixer(data_anom, data_norm, d):
    output = np.zeros_like(data_anom)
    output[:,:,:d] = data_anom[:,:,:d]
    output[:,:,d:] = data_norm[:,:,d:]
    return output

