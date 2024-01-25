from __future__ import division
import numpy as np
import scipy.special as sp
from scipy.integrate import quad
import scipy

# Question 1 functions

def complex_quadrature(func, a, b, i ,f):
    def real_func(x, i, f):
        return scipy.real(func(x, i, f))
    def imag_func(x, i, f):
        return scipy.imag(func(x, i, f))
    
    real_integral = quad(real_func, a, b, args = (i, f))
    imag_integral = quad(imag_func, a, b, args = (i, f))
    
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

def xJ1x(x):
    return x*sp.jv(1, x)

def xJ0x(x):
    return x*sp.jv(0, x)

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

def chebyClen(c, m, a, b, x):
    y = (x - 0.5*(b+a))/(0.5*(b-a))
    y2 = 2*y
    
    d, dd = 0.0, 0.0
    for j in range(m-1, 0, -1):
        tmp = d
        d = y2*d-dd+c[j]
        dd = tmp

    return y*d - dd + 0.5*c[0]

def chder(c, a, b):
    n = len(c)
    ret = np.zeros(n)
    ret[-2] = 2*(n-1)*c[n-1]
    
    for i in range(n-3, -1, -1):
        ret[i] = ret[i+2] + 2*(i+1)*c[i+1]
    
    con = 2/(b-a)

    return ret * con

def centered_dydx(func, x, delta):
    return (func(x+delta/2) - func(x-delta/2))/delta

def dydx(func, x, delta):
    return (func(x+delta) - func(x))/delta

# Question 2 functions

def sinPix(x):
    return np.sin(np.pi*x)

# Question 3 functions

def f(x):
    return np.exp(x)

class G():
    def __init__(self, delta):
        self.delta = delta

    def change_delta(self , new_delta):
        self.delta = new_delta

    def g(self, x):
        return 1/(x**2 + self.delta**2)

class H():
    def __init__(self, delta):
        self.delta = delta

    def change_delta(self, new_delta):
        self.delta = new_delta

    def h(self, x):
        return 1/(np.sin(np.pi*x/2)**2 + self.delta**2)

def u(x):
    return np.exp(-np.abs(x))

def v(x):
    return np.sqrt(x+1.1)


def Cos(x, n, func):
    return func(x) * np.cos(x*np.pi*n)

def Exp(x, n , func):
    return func(x) * np.exp(-1j*np.pi*n*x)

def fourierFit(func, n, splitAt0 = False):
    if splitAt0:
        c = 1/2*np.array([complex_quadrature(Exp, -1, 0, i, func)[0] for i in range(-(n//4), n//4+1)] + [complex_quadrature(Exp, 0, 1, i, func)[0] for i in range(-(n//4), n//4+1)])        
    else:
        c = 1/2*np.array([complex_quadrature(Exp, -1, 1, i, func)[0] for i in range(-(n//2), n//2+1)])
    
    return c

def _fourierClen(C, x):
    n = len(C)
    b = np.zeros(n+2, dtype=np.complex128)

    for i in range(n-1, -1, -1):
        b[i] = C[i] + np.exp(1j*np.pi*x)*b[i+1]

    #print(b/np.exp(1j*np.pi*x*n//2))

    return b[0]/np.exp(1j*np.pi*x*(n//2))

def fourierClen(C, x):
    return np.vectorize(_fourierClen, excluded=(0,))(C, x)

def _fourierRaw(C, x):
    n = len(C)
    acc = 0.0
    for i in range(n):
        acc += C[i]*np.exp(1j*np.pi*x*i)

    return acc/np.exp(1j*np.pi*x*(n//2))


def fourierRaw(C, x):
    return np.vectorize(_fourierRaw, excluded=(0,))(C, x)
