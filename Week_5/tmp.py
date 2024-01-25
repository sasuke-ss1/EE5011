from __future__ import division
import scipy.special as sp
import numpy as np

def F(i,x):
    return sp.jv(i, x)


def s(x, N):
    y = [1/(1+i)*sp.jv(i, x) for i in range(N+1)]
    return sum(y)


def recur():
    J = [0, 1] #0, 1
    J_ = [1, 0]
    x =15
    for i in range(1, 41):
        J.append(20/15*J[i] - J[i-1])
        J_.append(20/15*J_[i] - J_[i-1])

    J = np.array(J[2:])
    J_ = np.array(J_[2:])

    print(np.mean(np.abs(J - J_)))

#recur()


def forward(x = 15):    
    J = np.zeros(41)
    J[0] = sp.jv(0, x)
    J[1] = sp.jv(1, x)

    for i in range(1, 40):
        J[i+1] = 2*i/x*J[i] - J[i-1]

    return J

import matplotlib.pyplot as plt
x = 15

cal = forward(x)
exact = [sp.jv(i, x) for i in range(41)]

backward= [0, 1]
for i in range(60):
    backward.append(2*(60 - i)*backward[-1]/x - backward[-2])
    
backward = [backward[i]/backward[-1] for i in range(len(backward))][::-1][:41]

plt.figure()
plt.semilogy(np.abs(cal), label="Forward")
#plt.semilogy(np.abs(cal), label="Forward")
plt.semilogy(np.abs(exact), label = "Exact")
plt.semilogy(np.abs(backward), label = "Backward")
plt.legend()
plt.show()


