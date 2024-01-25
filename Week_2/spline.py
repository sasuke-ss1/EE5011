import scipy as sp
import scipy.special as sc
import matplotlib.pyplot as plt
import weave as wv
import numpy as np

# Function
def f(x):
    return x**(1+sc.j0(x))/np.sqrt((1+x**2*100)*(1-x))

def df1(x):
    return x**(1+sc.j0(x))*(1+sc.j0(x))/x - sc.j1(x)*np.log(x)*x**(1+sc.j0(x))

def f1(x):
    return x**(1+sc.j0(x))

def f2(x):
    return np.sqrt((1-x)*(1+100*x**2))

def df2(x):
    return (200*x - 300*x**2 - 1)*(2*np.sqrt((1-x)*(1+100*x**2)))

def df(x):
    ret = f2(x)* df1(x) - f1(x) * df2(x)

    return ret/f2(x)**2

# Questions

def make_table(func, name):
    x = np.arange(0.1, 0.90+0.05, 0.05)
    y = func(x)

    plt.figure(figsize=(10, 7))
    plt.plot(x, y)
    plt.xlabel(r'$x\rightarrow$')
    plt.ylabel(r"$y\rightarrow$")
    plt.title('f(x) vs x')    
    plt.savefig(name)

    return x, y

def spline(func, name, save_name):
    with open(name, 'r') as r:
        C = r.read()
    
    h = np.logspace(-4, -2, 20)
    N = 0.8/h
    err = np.zeros_like(h)

    plt.figure(0, (10, 7))

    for i in range(len(h)):
        x = np.linspace(0.1, 0.9, N[i])
        y = func(x)
        n = int(N[i])
        xx= np.linspace(0.1, 0.9, 10*n+1)
        y2, u, yy = np.zeros_like(x), np.zeros_like(x), np.zeros_like(xx)
        code = """
        #include <math.h>
	    int i;
	    double xp;
	    spline(x,y,n,0,0,y2,u);
	    for(i=0; i<=10*n; i++){
		    xp=xx[i];
		    splint(x,y,y2,n,xp,yy+i);
	    }
        
        """
        wv.inline(code,["x","y","n","y2","u","xx","yy"], support_code = C, extra_compile_args=["-g"], compiler="gcc")
        if not i:
            plt.figure(1, figsize=(10,7))
            plt.plot(x, y, 'ro')
            plt.plot(xx, yy, 'b')
            plt.title("Interpolated Values and data Points for n={}".format(N[i]))
            plt.savefig(save_name.split('.')[0] + '_idx.png')
            plt.clf()

        plt.figure(0)
        diff = np.abs(func(xx)- yy)
        err[i] = np.max(diff)
        
        plt.plot(xx, diff, label='N={}'.format(N[i]))
    plt.legend()
    plt.savefig(save_name.split('.')[0] + "_diff.png")
    plt.clf()

    plt.xlabel(r"$Spacing\rightarrow$")
    plt.ylabel(r"$Error\rightarrow$")
    plt.figure(2, figsize=(10,7))
    plt.loglog(h, err)
    plt.grid()
    plt.xlabel(r"$Spacing\rightarrow$")
    plt.ylabel(r"$Error\rightarrow$")
    plt.title("Error vs Spacing")
    plt.savefig(save_name)
    plt.clf()

if __name__ == "__main__":
    make_table(f, 'fx.png')
    spline(f, 'spline.c', 'org.png')
    spline(f, 'spline_nak.c', 'nak.png')
    #make_table(df, 'dfx.png')
