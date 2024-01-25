from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
from scipy.integrate import quad

def s(x, N):
    y = [1/(1+i)*sp.jv(i, x) for i in range(N+1)]
    return sum(y)

def c(i):
    return 1/(1+i)

def F(i,x):
    return sp.jv(i, x)

def alpha(x,i):
    return 2*(i+1)/x
    
def part1(name,  savename, labels=["Non Cancelling roots", "Near Cancellation roots"]):
    x = np.loadtxt(name, usecols=0)
    y1 = np.loadtxt(name, usecols=1)
    y2 = np.loadtxt(name, usecols=2)

    x = np.delete(x,0,0)
    y1 = np.delete(y1,0,0)
    y2 = np.delete(y2,0,0)

    
    plt.figure(figsize=(10, 7))
    plt.loglog(x[::100], y1[::100], label=labels[0])
    plt.loglog(x[::100], y2[::100], label=labels[1])
    plt.legend()
    plt.xlabel(r"$Alpha\rightarrow$")
    plt.ylabel(r"$Error\rightarrow$")
    plt.savefig(savename)

def part2():
    # Forward
    x = 15
    forward, forward10 = [0, sp.jv(0,x)], [0, sp.jv(0, x/10)] # Check
    for i in range(40 + 1):
        forward.append(2*i/x*forward[-1] - forward[-2])
        forward10.append(2*i/(x/10)*forward10[-1] - forward10[-2])

    acc, acc10, errors, errors10 = 0, 0, [], []
    N = range(41)    
    
    for n in N:
        acc += 1/(n+1)*forward[n+1]   
        acc10 += 1/(n+1)*forward10[n+1]
        
        errors.append(np.abs(s(x, n) - acc))
        errors10.append(np.abs(s(x/10, n) - acc10))

    plt.figure(figsize=(10, 7))
    plt.semilogy(N, errors, label=r"$x=15$")
    plt.semilogy(N, errors10, label=r"$x=1.5$")
    plt.xlabel(r'$n\rightarrow$')
    plt.ylabel(r'$Error\rightarrow$')
    plt.legend()
    plt.title('Error plot')
    plt.savefig('Forward.png')

    backward, backward10 = [0, 1], [0 ,1]
    for i in range(60):
        backward.append(2*(60 - i)*backward[-1]/x - backward[-2])
        backward10.append(2*(60 - i)*backward10[-1]/(x/10) - backward10[-2])

    backward = [backward[i]/backward[-1] for i in range(len(backward))][::-1][:41]
    backward10 = [backward10[i]/backward10[-1] for i in range(len(backward10))][::-1][:41]
    
    errors10, errors, acc, acc10 = [], [], 0, 0
    for i in range(41): # Check
        acc += backward[i]/(1 + i)
        acc10 += backward10[i]/(1 + i)
        errors.append(abs(s(x, i) - acc))
        errors10.append(abs(s(x/10, i) - acc10))
    
    xval = list(range(41))

    plt.figure(figsize=(10, 7))
    plt.semilogy(xval, errors, label=r"$x=15$")
    plt.semilogy(xval, errors10, label=r"$x=1.5$") 
    plt.xlabel(r'$n\rightarrow$')
    plt.ylabel(r'$Error\rightarrow$')
    plt.legend()
    plt.title('Error plot')
    plt.savefig('Backward.png')

def part3(func, a, n, alpha, beta): 
    b = np.zeros(n+2)
    
    for i in range(n-1, -1, -1):
        b[i] = a(i) + alpha(i)*b[i+1] + beta(i+1)*b[i+2]

    return func(0)*a(0) + func(1)*b[1] + beta(1)*func(0) + b[2]

def chebyfit(n, a, b, func):
    '''
    Calculates all the Chebyshev Coefficients
    '''
    bma = (b-a)/2
    bpa = (b+a)/2
    tmp, ret = [], []

    for i in range(n):
        y = np.cos(np.pi*(i+0.5)/n)
        tmp.append(func(y*bma + bpa))
    
    fac = 2/n
    for i in range(n):
        acc = 0
        for j in range(n):
            acc += tmp[j] * np.cos(np.pi*i*(j+0.5)/n)
        
        ret.append(fac*acc)
    
    return np.array(ret)


def chebyRaw(c, m, a, b, x):
    ret = 0.0

    y = (x - 0.5*(b+a))/(0.5*(b-a))
    for i, coeff in enumerate(c):
        ret += coeff*np.cos(i*np.arccos(y))

    return ret -c[0]/2

def chebyClen(c, m, a, b, x):
    y = (x - 0.5*(b+a))/(0.5*(b-a))
    y2 = 2*y
    
    d, dd = 0.0, 0.0
    for j in range(m-1, 0, -1):
        tmp = d
        d = y2*d-dd+c[j]
        dd = tmp

    return y*d - dd + 0.5*c[0]

def fourierFit(func, n):
    a0 = quad(func, -1, 1)[0]
    an = np.array([quad(Cos, -1, 1, args=(i, func))[0] for i in range(1, n+1)])
    bn = np.array([quad(Sin, -1, 1, args=(i, func))[0] for i in range(1, n+1)])
        
    return a0, an, bn

def Cos(x, n, func):
    return func(x) * np.cos(x*np.pi*n)

def Sin(x, n, func):
    return func(x) * np.sin(x*np.pi*n)

def exp(x):
    return np.exp(x)

def fourierRaw(a0, an, bn, x):
    ret = 0.5*a0

    cosine_sum, sine_sum = 0.0, 0.0
    for i in range(len(an)):
        cosine_sum += an[i]*np.cos((i+1)*np.pi*x)
        sine_sum += bn[i]*np.sin((i+1)*np.pi*x)

    return ret + cosine_sum + sine_sum

def fourierClen(a0, an, bn, x):
    theta = np.pi*x
    
    d, dd = 0.0, 0.0
    n = len(an)
    
    for i in range(n, 0, -1):
        tmp = d
        d = bn[i-1] + 2*np.cos(theta)*d - dd 
        dd = tmp

    sine_sum = d*np.sin(theta)

    d, dd= 0.0, 0.0

    for i in range(n, 0, -1):
        tmp = d
        d = an[i-1] + 2*np.cos(theta)*d - dd
        dd = tmp

    cosine_sum = 0.5*a0 + np.cos(theta)*d -dd 

    return cosine_sum + sine_sum

if __name__ == '__main__':
    part1("exact.txt", 'Exact.png')
    part1("accurate.txt", 'Accurate.png', ["Near Cancellation roots", "Non Cancelling roots"])
    part2()

    x = np.linspace(-0.9, 0.9, num=100)
    
    coeff = chebyfit(100, -1, 1, exp)

    #rawfunc = np.vectorize(chebyRaw, excluded=[0])
    #clenfunc = np.vectorize(chebyClen, excluded=[0])
    raw = chebyRaw(coeff, 100, -1, 1, x)
    clen = chebyClen(coeff, 100, -1, 1, x)
    exact = exp(x)
    error_raw =np.abs(raw - exact)
    error_clen = np.abs(clen - exact)

    print("Mean error using raw implementaion:", np.mean(error_raw))
    print("Mean error using clenshaw summation:", np.mean(error_clen))

    print("Std. of Error using raw implementaion:", np.std(error_raw))
    print("Std. of Error using clenshaw summation:", np.std(error_clen))

    a0, an, bn = fourierFit(exp, 10)

    x = np.linspace(-0.9, 0.9, 100)
    raw = fourierRaw(a0, an, bn, x)
    clen = fourierClen(a0, an, bn, x)

    error_raw = np.abs(raw - exp(x))**2
    error_clen = np.abs(raw - exp(x))**2

    print("Mean error using raw implementation:", np.mean(error_raw))
    print("Mean error using clenshaw summation:", np.mean(error_clen))

    print("Std. of Error using raw implementation:", np.std(error_raw))
    print("Std. of Error using clenshaw summation:", np.std(error_clen))


    errors = []
    for i in range(5, 100):
        a0, an, bn = fourierFit(exp, i)

        x = np.linspace(-0.9, 0.9, 100)
        raw = fourierRaw(a0, an, bn, x)
        clen = fourierClen(a0, an, bn, x)

        error_raw = np.max(np.abs(raw - exp(x)))
        error_clen = np.max(np.abs(raw - exp(x)))

        errors.append(error_clen)

    plt.plot(errors)
    plt.show()        
