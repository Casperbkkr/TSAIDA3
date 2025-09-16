import numpy as np



def rel_contrast(dist_profile1):
    #dist_profile1 = (dist_profile1 - np.mean(dist_profile1)) / dist_profile1.std()
    max1 = np.max(dist_profile1[1:])
    min1 = np.min(dist_profile1[1:])

    rel1 = (max1 - min1) / min1

    return rel1