from scipy import integrate, interpolate
import scipy.special as sp
import numpy as np
import matplotlib.pyplot as plt
import romberg

count = 0
def f(x):
    global count
    count += 1
    if x < 1:
        return 2*sp.jv(3, 2.7*x)**2*x
    return 2*abs(sp.jv(3, 2.7)/sp.kv(3, 1.2))**2*sp.kv(3, 1.2*x)**2*x

def exact():
    return sp.jv(3, 2.7)**2 - sp.jv(4, 2.7)*sp.jv(2, 2.7) + abs(sp.jv(3, 2.7)/sp.kv(3, 1.2))**2*(sp.kv(4, 1.2)*sp.kv(2, 1.2) - sp.kv(3, 1.2)**2)

def qrombPy(f, a, b, e=1e-6, k=5):
    h, yy, y, outT = [], [], [], 0
    for i in range(1, 21):
        h.append(20/4**(i - 1))
        outT = romberg.trapzd(f, a, b, outT, i)
        yy.append(outT)
        
        if i >= k:
            out, err = romberg.polint(h[-k:], yy[-k:], 0)
            y.append(out)
            
            if abs(err) <= e*abs(out):
                break

    return h[k - 1 :], y

def trapzdBy3(f, a, b, n):
    if n == 1:
        return 1/4*(b - a)*(f((3*a+b)/2) + f((a+3*b)/2) + f(a+b)/2)

    else:
        d = (float)(b - a)/3**(n - 1)
        x, out = a + d, 0
        while x < b:
            out += f(x)*d
            x += d
        return out + 0.25*d*(f(a) + f((a+b)/2) + f(b))
        
def qrombBy3(f, a, b, eps=1e-6, k=5):
    h, yy, y = [], [], []
    for i in range(1, 21):
        h.append(20 / 9 ** (i - 1))
        yy.append(trapzdBy3(f, a, b, i))
        if i >= k:
            out, err = romberg.polint(h[-k:], yy[-k:], 0)
            y.append(out)
            if abs(err) <= eps * abs(out):
                break

    return h[k - 1 :], y

x = np.linspace(10, 0, 200, endpoint=False)
vectorized_f = np.vectorize(f)

plt.figure(figsize=(10, 7))
plt.title("Integrand vs x")
plt.semilogy(x, vectorized_f(x))
plt.xlabel(r"$x\rightarrow$")
plt.ylabel(r"$f(x)\rightarrow$")
plt.savefig("imgs/integrand.png")
plt.close()

counts, errors = [], []
for i in range(5, 30):
    y_pred, _, info = integrate.quad(f, 0, i, full_output=True)
    errors.append(np.abs(y_pred - exact()))
    counts.append(info["neval"])

fig, axs = plt.subplots(2, 1)
axs[0].set_ylabel(r"Error$\rightarrow$")
axs[0].set_title(r"scipy Quad(f, 0, $x$)")
axs[0].set_xticks([])
axs[0].semilogy(range(5, 30), errors)

axs[1].set_title("Scipy Quad efficiency")
axs[1].set_xlabel(r"$x\rightarrow$")
axs[1].set_ylabel(r"Num Calls$\rightarrow$")
axs[1].plot(range(5, 30), counts)

plt.savefig('imgs/scipy_quad.png')
plt.close()

y_pred = 0
countsTrapzd, errorsTrapzd = [], []
count = 0
for i in range(1, 20):
    y_pred = romberg.trapzd(f, 0, 20, y_pred, i)
    countsTrapzd.append(count)
    errorsTrapzd.append(np.abs(y_pred - exact()))
    count = 0

plt.figure(figsize=(10, 7))
plt.title(r"Error vs $Spacing$ (Trapzd)")
plt.loglog(20/2**(np.arange(1.0, 20.0) - 2), errorsTrapzd)
plt.xlabel(r"$Spacing\rightarrow$")
plt.ylabel(r"Error$\rightarrow$")
plt.savefig("imgs/h_trapzd.png")
plt.close()

plt.figure(figsize=(10, 7))
plt.loglog(countsTrapzd, errorsTrapzd)
plt.title("Error Efficiency in Trapzd")
plt.xlabel(r"Num Calls$\rightarrow$")
plt.ylabel(r"Error$\rightarrow$")
plt.savefig("imgs/error_trapzd.png")
plt.close()

yL = 0;yR = 0
countsTrapzdSplit, errorsTrapzSplit = [], []
for i in range(2, 20):
    count = 0
    yL = romberg.trapzd(f, 0, 1, yL, i // 2)
    yR = romberg.trapzd(f, 1, 20, yR, i - i//2)
    errorsTrapzSplit.append(np.abs(yL + yR - exact()))
    countsTrapzdSplit.append(count)

plt.figure(figsize=(10, 7))
plt.title("Error Efficiency (Trapzd Split)")
plt.xlabel(r"$h\rightarrow$")
plt.ylabel(r"Error$\rightarrow$")
plt.loglog(20 / 2 ** (np.arange(2.0, 20.0) - 2), errorsTrapzSplit)
plt.savefig("imgs/h_trapzd_split.png")
plt.close()

plt.figure(figsize=(10, 7))
plt.title("Error Efficiency (Trapzd Split)")
plt.xlabel("Num Calls")
plt.ylabel(r"Error$\rightarrow$")
plt.loglog(countsTrapzdSplit, errorsTrapzSplit)
plt.savefig("imgs/error_trapzd_split.png")
plt.close()

count = 0
counts_qromb, errors_qromb = [], []
for i in range(1, 14):
    errors_qromb.append(np.abs(romberg.qromb(f, 0, 20, 10**-i)[0] - exact()))
    counts_qromb.append(count)
    count = 0

plt.figure(figsize=(10, 7))
plt.title("Error Efficiency (Qromb)")
plt.xlabel(r"Num Calls$\rightarrow$")
plt.ylabel(r"Error$\rightarrow$")
plt.loglog(counts_qromb, errors_qromb)
plt.savefig("imgs/error_qromb.png")
plt.close()

counts_qromb_split, errors_qromb_split = [], []
count = 0
for i in range(1, 14):
    yL = romberg.qromb(f, 0, 1, 10**-i)[0]
    yR = romberg.qromb(f, 1, 20, 10**-i)[0]
    errors_qromb_split.append(np.abs(yL + yR - exact()))
    counts_qromb_split.append(count)
    count = 0

plt.figure(figsize=(10, 7))
plt.title("Error Efficiency (Qromb Split)")
plt.xlabel(r"Num Calls\rightarrow")
plt.ylabel(r"Error$\rightarrow$")
plt.loglog(counts_qromb_split, errors_qromb_split)
plt.savefig("imgs/qromb_split_error.png")
plt.close()

counts_py, errors_py = [], []
count = 0
for i in range(1, 10):
    errors_py.append(np.abs(qrombPy(f, 0, 20, 10**-i)[1][-1] - exact()))
    counts_py.append(count)
    count = 0

plt.figure(figsize=(10, 7))
plt.title("Error Efficiency (Python Qromb)")
plt.xlabel(r"Num Calls$\rightarrow$")
plt.ylabel(r"Error$\rightarrow$")
plt.loglog(counts_py, errors_py)
plt.savefig("imgs/error_py.png")
plt.close()

counts_py_split, errors_py_split = [], []
count = 0
for i in range(1, 10):
    y = qrombPy(f, 0, 1, 10**-i)[1][-1] + qrombPy(f, 1, 20, 10**-i)[1][-1]
    errors_py_split.append(np.abs(y - exact()))
    counts_py_split.append(count)
    count = 0

plt.figure(figsize=(10, 7))
plt.title("Error Efficiency (Python Qromb Split)")
plt.xlabel(r"Num Calls$\rightarrow$")
plt.ylabel(r"Error$\rightarrow$")
plt.loglog(counts_py_split, errors_py_split)
plt.savefig("imgs/error_py_split.png")
plt.close()

countsQrombOrder = []
for i in range(3, 21):
    count = 0
    romberg.qromb(f, 0, 20, 1e-8, i)
    countsQrombOrder.append(count)

plt.figure(figsize=(10, 7))
plt.title(r"Num Calls with Fixed Error (Qromb)")
plt.xlabel(r"Order\rightarrow")
plt.ylabel(r"Num Calls$\rightarrow$")
plt.semilogy(range(3, 21), countsQrombOrder)
plt.savefig("imgs/calls_qromb.png")
plt.close()

errorsSplint = []
for i in range(3, 20):
    x = np.linspace(0, 20, 2**i)
    y = interpolate.splint(0, 20, interpolate.splrep(x, vectorized_f(x)))
    errorsSplint.append(np.abs(y - exact()))

plt.figure(figsize=(10, 7))
plt.title("Error Efficiency (Splint)")
plt.xlabel(r"Num Calls$\rightarrow$")
plt.ylabel(r"Error$\rightarrow$")
plt.loglog(2**np.arange(3, 20), errorsSplint)
plt.savefig("imgs/error_splint.png")
plt.close()

errorsSplintSplit = []
for i in range(3, 20):
    x_lt_1 = np.linspace(0, 1, 2 ** (i - 1))
    x_gt_1 = np.linspace(1, 20, 2 ** (i - 1))
    tmp_lt_1 = interpolate.splrep(x_lt_1, vectorized_f(x_lt_1))
    tmp_gt_1 = interpolate.splrep(x_gt_1, vectorized_f(x_gt_1))
    y = interpolate.splint(0, 1, tmp_lt_1) + interpolate.splint(1, 20, tmp_gt_1)
    errorsSplintSplit.append(np.abs(y - exact()))

plt.figure(figsize=(10, 7))
plt.title("Error Efficiency (Splint Split)")
plt.xlabel(r"Num Calls$\rightarrow$")
plt.ylabel(r"Error$\rightarrow$")
plt.loglog(2 ** np.arange(3, 20), errorsSplintSplit)
plt.savefig("imgs/error_splint_split.png")
plt.close()

h, y = qrombBy3(f, 0, 20)
plt.figure(figsize=(10, 7))
plt.title(r"Error vs $h$ (Python Qromb)")
plt.ylabel(r"Error\rightarrow")
plt.xlabel(r"$h$\rightarrow")
plt.loglog(h, np.abs(y - exact()))
plt.savefig("imgs/h_error_By3_py_.png")
plt.close()

countsBy3, errorsBy3 = [], []
count = 0
for i in range(1, 10):
    errorsBy3.append(np.abs(qrombBy3(f, 0, 20, 10**-i)[1][-1] - exact()))
    countsBy3.append(count)
    count = 0

plt.figure(figsize=(10, 7))
plt.title("Error Efficiency (Python Qromb)")
plt.xlabel(r"Num Calls$\rightarrow$")
plt.ylabel(r"Error$\rightarrow$")
plt.loglog(countsBy3, errorsBy3)
plt.savefig("imgs/error_By3_py.png")
plt.close()

countsBy3Split, errorsBy3Split = [], []
count = 0
for i in range(1, 10):
    y = qrombBy3(f, 0, 1, 10**-i)[1][-1] + qrombBy3(f, 1, 20, 10**-i)[1][-1]
    errorsBy3Split.append(np.abs(y - exact()))
    countsBy3Split.append(count)
    count = 0

plt.figure(figsize=(10, 7))
plt.title("Error Efficiency (Python Qromb Split)")
plt.xlabel(r"Num Calls$\rightarrow$")
plt.ylabel(r"Error$\rightarrow$")
plt.loglog(countsBy3Split, errorsBy3Split)
plt.savefig("imgs/error_By3_py_split.png")
plt.close()

plt.figure(figsize=(10, 7))
plt.title("Error Efficiency")
plt.xlabel(r"Num Calls$\rightarrow$")
plt.ylabel(r"Error$\rightarrow$")
plt.loglog(countsTrapzd, errorsTrapzd, label="Trapzd")
plt.loglog(countsTrapzdSplit, errorsTrapzSplit, label="Trapzd Split")
plt.loglog(counts_qromb, errors_qromb, label="Qromb")
plt.loglog(counts_qromb_split, errors_qromb_split, label="Qromb Split")
plt.loglog(counts_py, errors_py, label="Python Qromb")
plt.loglog(counts_py_split, errors_py_split, label="Python Qromb Split")
plt.loglog(2 ** np.arange(3, 20), errorsSplint, label="Splint")
plt.loglog(2 ** np.arange(3, 20), errorsSplintSplit, label="Splint Split")
plt.loglog(countsBy3, errorsBy3, label="Python Qromb 3")
plt.loglog(countsBy3Split, errorsBy3Split, label="Python Qromb 3 Split")
plt.legend()
plt.savefig("imgs/all.png")
plt.close()