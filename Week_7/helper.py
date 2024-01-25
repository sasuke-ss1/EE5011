from __future__ import division
import numpy as np


def BoxMuller(n):
    ret = []
    for _ in range(n):
        rsq = 0
        while (rsq == 0 or rsq >= 1):
            x = np.random.uniform(0, 1, 2)
            v = 2*x - 1
            rsq = np.sum(v**2)

        ret.append(v[0] * np.sqrt(-2*np.log(rsq)/rsq))            

    return np.array(ret)

def Dist(n):
    ret = []
    for _ in range(int(0.999*n)):
        rsq = 0
        while (rsq == 0 or rsq >= 1):
            x = np.random.uniform(0, 1, 2)
            v = 2*x - 1
            rsq = np.sum(v**2)

        ret.append(v[0] * np.sqrt(-2*np.log(rsq)/rsq))            

    ret.append(500*v[0]*np.sqrt(-2*np.log(rsq)/rsq))

    return np.array(ret)
    
