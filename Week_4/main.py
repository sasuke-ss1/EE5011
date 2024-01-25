from functions import *
import matplotlib.pyplot as plt
import gauss_quad as gq
import romberg as r

# Question 1
x1 = np.linspace(0, 1, num=200)
x2 = np.linspace(1, 15, num=200)

plt.figure(figsize=(10, 7))
plt.loglog(x1, f1(x1))
plt.xlabel(r"$x\rightarrow$")
plt.ylabel(r"$y\rightarrow$")
plt.savefig("imgs/f1.png")
plt.close()

plt.figure(figsize=(10, 7))
plt.semilogy(x2, f2(x2))
plt.xlabel(r"$x\rightarrow$")
plt.ylabel(r"$y\rightarrow$")
plt.savefig("imgs/f2.png")
plt.close()

# Question 2
val1, error, dic = quad(f1, 0, 1, epsabs=1e-12, epsrel=1e-12, full_output=True)
print("Number of function evaluation of f1(x) required for {} is {}".format(error, dic["neval"]))

val2, error, dic = quad(f2, 1, np.inf, epsabs=1e-12, epsrel=1e-12, full_output=True)
print("Number of function evaluation of f2(x) required for {} is {}".format(error, dic["neval"]))

# Question 3
x, w = gq.gauleg(0, 1, 7)
acc = 0.0

for i in range(len(x)):
    acc += f1(x[i]) * w[i]

print("Gauss-Legendre requires 10 f1(x) calls to get an error of {}".format(np.abs(acc - val1)))

x, w = gq.gaulag(30, 0.0)
acc = 0.0
for i in range(len(x)):
    acc += approxf2(x[i]) * w[i]

print("Gauss-Laguerre requires 100 f2(x) calls to get an error of {}".format(np.abs(acc - val2))) # Check

# Question 4
val1R = r.qromb(f1, 0, 1, eps=1e-12)

print("Romberg Integration required {} f1(x) evaluation to get an error of {}".format(val1R[-1], abs(val1R[1])))

# Question 5
val2R = r.qromb(transformedf2, np.pi/4, np.pi/2, eps=1e-12)

print("Romberg Integration required {} f2(x) evaluation to get an error of {}".format(val2R[-1], abs(val2R[1])))

# Question 7
x = np.linspace(1 + 1e-6, 3, num=100, endpoint=False)
t = np.linspace(-1 + 1e-6, 1, num=100, endpoint=False)

plt.figure(figsize=(10, 7))
plt.semilogy(x, dJ(x))
plt.xlabel(r"$x\rightarrow$")
plt.ylabel(r"$y\rightarrow$")
plt.savefig("imgs/dJ.png")
plt.close()

plt.figure(figsize=(10, 7))
plt.semilogy(t, tranfromedJ(t))
plt.xlabel(r"$x\rightarrow$")
plt.ylabel(r"$y\rightarrow$")
plt.savefig("imgs/TdJ.png")
plt.close()

# Question 8 CHECK 
N=20
j = np.arange(1, N+1)
x = X(j, N)
w = W(j, N)

exact = calcGQ(cheby, x, w)

errors = []
for i in range(1, N+1):
    j = np.arange(1, i+1)
    x = X(j, i)
    w = W(j, i)

    val = calcGQ(cheby, x, w)
    errors.append(np.abs(val - exact))
    
plt.figure(figsize=(10, 7))
plt.semilogy(range(1, N+1), errors)
plt.xlabel(r"$N\rightarrow$")
plt.ylabel(r"Absolute Error$\rightarrow$")
plt.savefig("imgs/ErrorvsN.png")
plt.close()