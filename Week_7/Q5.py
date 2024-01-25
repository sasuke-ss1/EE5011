from __future__ import division
import numpy as np
import scipy.special as sp
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.integrate import quad

def u(x,y):
    a = np.pi * np.sin(10*(np.sqrt(x**2+y**2)-0.5))

    return x*np.cos(a) + y*np.sin(a) - 0.5

def v(x,y):
    a = np.pi * np.sin(10*(np.sqrt(x**2+y**2) - 0.5))

    return -x*np.sin(a) + y*np.cos(a) - 0.5

def func(x, y):
    return u(x, y)**2 + v(x, y)**2

def norm(x):
    return (x - x.min())/(x.max() - x.min())

if __name__ == '__main__':
    x, y = np.linspace(-5, 5, 1000), np.linspace(-5, 5, 1000)
    xgrid, ygrid = np.meshgrid(x, y)
    z = func(xgrid, ygrid)

    h, w = z.shape
    r = np.zeros_like(z)
    r[np.abs(z) < 1] = z[np.abs(z) < 1]
    

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contourf(xgrid, ygrid, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(ax.contourf(xgrid, ygrid, z))
    plt.savefig('./imgs/8.png')

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contourf(xgrid, ygrid, r)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(ax.contourf(xgrid, ygrid, r))
    plt.savefig('./imgs/9.png')

    fig, ax = plt.subplots(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    p = ax.contour3D(xgrid, ygrid, z, 50, cmap='GnBu')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.colorbar(p)
    plt.savefig('./imgs/10.png')

    invol, outvol = 0, 0
    for i in range(100000):
        x1, x2 = 4*np.random.rand(2) - 2   
        val = func(x1,x2)
        
        if(abs(val) < 1):
            invol += 1
        else:
            outvol += 1

    area = 16*(invol/(invol+outvol))
    print("Estimated Area is {:0.4f}".format(area))
