from __future__ import division
import numpy as np
import scipy.special as sp
from scipy.integrate import quad
import scipy as sc
from functions import *

def fun(x):
    return x*sp.jv(1, x)

def chebyfit(n, a, b, func):
    '''
    Calculates all the Chebyshev Coefficients
    '''
    bma = (b-a)/2
    bpa = (b+a)/2
    tmp, ret = [], []

    for i in range(1, n+1):
        y = np.cos(np.pi*(i-0.5)/n)
        tmp.append(func(y*bma + bpa))
    
    for i in range(n):
        acc = 0
        for j in range(1, n+1):
            acc += tmp[j-1] * np.cos(np.pi*i*(j-0.5)/n)
        
        ret.append(2/n*acc)
    
    return ret

def chebypol(n, x):
    '''
    Caculates Tn(x) by forward recursion

    '''
    if  n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return 2*x*chebypol(n-1, x) - chebypol(n-2, x)

def coeff2func(coeff, m, x, a, b):
    ret, y = -0.5*coeff[0], -1+(x+a)*2/(b-a)
    for i in range(m):
        ret += coeff[i] * chebypol(i, y)
    
    return ret

def error(x, true_func, est_func, coeff, d=None, a=0, b=5):
    if d:
        ret = []
        for i in range(len(x)):
            ret.append(np.abs(true_func(x[i]) - est_func(coeff, d, x[i], a, b)))
        return ret
    ret = []
    for i in range(len(x)):
        ret.append(np.abs(true_func(x[i]) - est_func(coeff, len(x), x[i], a, b)))

    return ret

def mapping(x, min_x, max_x, min_to, max_to):
    return (x-min_x)/(max_x-min_x) * (max_to-min_to) + min_to

def cheby_approx(coeff, m, x, a, b):
    T = [0, mapping(x, a, b, -1, 1), 1]
    ret = coeff[0]/2 + coeff[1]*T[1]
    x, i = 2*T[1], 2

    while i < m:
        T[0] = x*T[1] - T[2]
        ret += coeff[i]*T[0]
        T[2], T[1] = T[1], T[0]

        i += 1
    
    return ret

def dfun(x):
    return x*sp.jv(0, x)

def dcheby(coeff, m, a, b):
    coeff_der = np.zeros(m)
    coeff_der[-2] = 2*(m-1)*coeff[m-1]
    i = m - 3
    while i >= 0:
        coeff_der[i] = coeff_der[i+2]+2 * (i+1) * coeff[i+1]
        i -= 1

    return [i*2/(b-a) for i in coeff_der]

def dfuncCentered(coeff, c, x, a, b):
    return (fun(x+c/2) - fun(x-c/2))/c

def f(x):
    return np.exp(x)

def g(x):
    return 1/(x**2 + 3**2)

def h(x):
    return 1/(np.cos(np.pi*x/2)**2 + 3**2)

def u(x):
    return np.exp(-1*np.abs(x))

def v(x):
    return np.sqrt(x+1.1)

def fourierCoeff(func, N, isU=False):
    if isU:
        return np.array([quad(func, -1, 0, args=(i))[0] for i in range(N//2)] + [quad(func, 0, 1, args=(i))[0] for i in range(N//2)]) 
    
    return np.array([quad(func, -1, 1, args=(i))[0] for i in range(N)])

def fP(x, m):
    return f(x) * np.cos(m*(x+1)*np.pi/2)

def hP(x, m):
    return h(x) * np.cos(m*(x+1)*np.pi/2)

def gP(x, m):
    return g(x) * np.cos(m*(x+1)*np.pi/2)

def uP(x, m):
    return u(x) * np.cos(m*(x+1)*np.pi/2)

def vP(x, m):
    return v(x) * np.cos(m*(x+1)*np.pi/2)

def romberg(x):
    if x < 1:
        return 2*x*sp.jv(3, 2.7*x)**2
    else:
        return 2*x*abs(sp.jv(3, 2.7)/sp.kv(3, 1.2))**2*sp.kv(3, 1.2*x)**2

def eval(coeffs, m, x, a, b):
    y = (2.0 * x - a - b) * 1.0/(b - a)
    d, dd = coeffs[-1], 0
    for coef in coeffs[-2:0:-1]:
        d, dd = 2.0*y*d - dd +coef, d

    return y*d - dd + 0.5 * coeffs[0]

def eval_fourier(coeffs, m, x, a, b):
    d, dd = coeffs[-1], 0
    for coef in coeffs[-2:0:-1]:
        d, dd = 2.0*x*d - dd +coef, d
    
    return x*d - dd + 0.5 * coeffs[0]


def Cos(x):
    return np.cos(np.pi*x)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #C = np.abs(fourierFit(Cos, 2))
    #print(exactC)
    
    x = np.linspace(-1, 1, 100)

    #Approx = fourierClen(C, x)
    
    #Acc = exactC[0]*np.exp(-1j*np.pi*x) + exactC[1] + exactC[2]*np.exp(1j*np.pi*x)
    #exact = Cos(x)
    #Raw = fourierRaw(C, x)

    #plt.figure()
    #plt.plot(x, Cos(x), label="Actual")
    #plt.plot(x, Raw, label="Approx")
    #plt.legend()
    #plt.savefig("tmp.png")

    y3 = G(3)
    y1 = G(1)
    y03 = G(0.3)
    
    plt.figure()
    plt.semilogy(x, y3.g(x), label="3")
    plt.semilogy(x, y1.g(x), label='1')
    plt.semilogy(x, y03.g(x), label="0.3")
    plt.legend()
    plt.savefig('tmp1.png')
