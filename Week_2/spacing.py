import matplotlib.pyplot as plt
import numpy as np
import math as m
import scipy.special as sp
import spline as s
import sys

assert len(sys.argv) == 6, "Enter Correct Spacing"

def deri1(x):
    y=-(((x**sp.j0(x))*(-100*x**3 + 2*(100*x**3 - 100*x**2 + x-1)*sp.j0(x) - (2*(100*x**3 - 100*x**2 + x-1)*x*m.log10(x)*sp.j1(x)) +x-2))/(2*(-100*x**3 + 100*x**2 -x +1)**1.5))
    
    return y

def func(x):
    y1=x**(1+sp.j0(x))/(np.sqrt(1-x+100*x**2-100*x**3))
    
    return y1


N1, N2, N3, N4, N5 = [int(i) for i in sys.argv[1:]]

x=list(np.linspace(0.1,0.2,N1, endpoint=False))
x2=list(np.linspace(0.2,0.4,N2, endpoint=False))

x3=list(np.linspace(0.4,0.6,N3, endpoint=False))
x4=list(np.linspace(0.6,0.8,N4, endpoint=False))

x5=list(np.linspace(0.8,0.9,N5))
x=x+x2+x3+x4+x5
y=[]

for i in range(len(x)):
    y.append(func(x[i]))
y2a=[0]*len(y)

y2a=s.spline(x,y,deri1(0.1),deri1(0.9))

xx=list(np.linspace(0.1,0.9,100))
yt=[]

for i in range(len(xx)):
    yt.append(func(xx[i]))

yy=s.splintn(x,y,y2a,xx)
err=max(abs(yy-yt))

plt.figure(figsize=(10, 7))
plt.grid()
plt.semilogy(xx,abs(yy-yt),"b")
plt.xlabel("Sample points")
plt.ylabel("Order of error")
plt.savefig("spacing/error_{}_{}_{}_{}_{}.png".format(N1, N2, N3, N4, N5))