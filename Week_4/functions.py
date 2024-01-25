from __future__ import division
import numpy as np
import scipy.special as sp
from scipy.integrate import quad

# Question 1
def f1(u):
    return u*sp.jv(3,2.7*u)**2 # Check

def f2(u):
    return u*sp.kv(3, 1.2*u)**2 

def f3(u):
    return u*sp.kv(3, u)**2

# Question 3

def sanity(x):
    return np.pi / (2.4)**2 * np.exp(-(x+2.4))

def approxf2(z):
    return f3(z/2 + 1.2)*np.exp(z)/(2*(1.2**2))


# Question 5
def transformedf2(w):
    return np.tan(w) * (1/np.cos(w)**2) * sp.kv(3, 1.2*np.tan(w))**2 

#Question 7
def dJ(x):
    arg = np.sqrt(-(x**2) + 4*x - 3)
    
    return np.exp(-x) / sp.jv(1, arg)

def tranfromedJ(x):
    arg = np.sqrt(1-x**2)
    
    return np.exp(-(x+2)) / sp.jv(1, arg)

def cheby(x):
    arg = np.sqrt(1-x**2)
    
    return np.exp(-(x+2)) / sp.jv(1, arg) * arg


# Question 8
def X(j, N):
    return np.cos(np.pi * (j - 0.5) / N)

def W(j, N):
    return np.full(len(j), np.pi / N)

def calcGQ(f, x, w):
    return np.sum(f(x)*w)