from functions import *
import matplotlib.pyplot as plt


h = H(1)
g = G(1)

def question1():
    x = np.linspace(0, 5, num=50)
    
    plt.figure(figsize=(10, 7))
    plt.plot(x, xJ1x(x))
    plt.xlabel(r"$x\rightarrow$")
    plt.ylabel(r"$y\rightarrow$")
    plt.title(r"$xJ_1(x)$")
    plt.savefig('./imgs/xj1x.png')
    plt.close()

    coeff = chebyfit(50, 0, 5, xJ1x)

    plt.figure(figsize=(10, 7))
    plt.semilogy(range(50), np.abs(coeff))
    plt.title("Magnitude plot")
    plt.xlabel(r"$x\rightarrow$")
    plt.ylabel(r"$Mag\rightarrow$")
    plt.savefig("./imgs/CoeffMag.png")
    plt.close()

    m = 20 # From graph
    
    chebyApprox = chebyClen(coeff, m, 0, 5, x)
    print("Maximum Absolute error in function estimate:", np.max(np.abs(chebyApprox - xJ1x(x))))

    dcoeff = chder(coeff, 0, 5)
    dchebyApprox = chebyClen(dcoeff, m, 0, 5, x)
    print("Maximum Absolute error in derivative estimate:", np.max(np.abs(dchebyApprox - xJ0x(x))))

    delta = np.logspace(-1, -8, 8, endpoint=True)
    
    centered_error, error = [], []
    for d in delta:
        ce = np.max(np.abs(xJ0x(x) - centered_dydx(xJ1x, x, d)))
        e = np.max(np.abs(xJ0x(x) - dydx(xJ1x, x, d)))
        centered_error.append(ce)
        error.append(e)
    
    plt.figure(figsize=(10, 7))
    plt.title('Centered Vs Uncentered')
    plt.loglog(delta, centered_error, label="Centered Derivative Error")
    plt.loglog(delta, error, label="Uncentered Derivative Error")
    plt.xlabel(r"$\Delta \rightarrow$")
    plt.ylabel(r"$Error\rightarrow$")
    plt.legend()
    plt.savefig('./imgs/DerivativeComparison.png')
    plt.close()

def question2():
    x = np.linspace(1, -1, num=50, endpoint=False)[::-1]
    coeff = chebyfit(40, -1, 1, sinPix)
    chebyApprox = chebyClen(coeff, 20, -1, 1, x)

    #plt.semilogy(range(40), np.abs(coeff))
    #plt.show()

    plt.figure(figsize=(10, 7))
    plt.title(r"Chebyshev Approx. of $sin(\pi x)$")
    plt.xlabel(r"$x\rightarrow$")
    plt.ylabel(r"$y\rightarrow$")
    plt.plot(x, sinPix(x), label="True Value")
    plt.plot(x, chebyApprox, label="Approx. Value")
    plt.legend()
    plt.savefig("./imgs/sinPix.png")
    plt.close()


    plt.figure(figsize=(10, 7))
    plt.title("Error in Estimate")
    plt.plot(x, np.abs(sinPix(x) - chebyApprox))
    plt.xlabel(r"$x\rightarrow$")
    plt.ylabel(r"$Error\rightarrow$")
    plt.savefig("./imgs/sinPixError.png")
    plt.close()

def question3():
    x = np.linspace(-1, 1, 50)
    coeffs_cheby = {
                "f_coeff": [chebyfit(50, -1, 1, f), f(x)],
                "g_coeff": [chebyfit(50, -1, 1, g.g), g.g(x)],
                "h_coeff": [chebyfit(50, -1, 1, h.h), h.h(x)],
                "u_coeff": [chebyfit(50, -1, 1, u), u(x)],
                "u\_break_coeff": [chebyfit(25, -1, 0, u) + chebyfit(25, 0, 1, u), u(x)],
                "v_coeff": [chebyfit(50, -1, 1, v), v(x)]
    }

    h.change_delta(3)
    coeff = chebyfit(50, -1, 1, h.h)
    plt.semilogy(range(len(coeff)), np.abs(coeff))
    #plt.semilogy(range(len(coeffs_cheby['u_coeff'][0])), np.abs(coeffs_cheby['u_coeff'][0]))
    plt.savefig("tmp.png")
    exit(0)

    N = range(5, 50, 5)
    for key, value in coeffs_cheby.items():
        name = key.split("_coeff")[0]
        errors = []
        for n in N:
            if name == "u\_break":
                x_neg = x[x<0]
                x_pos = x[x>0]
                
                chebyApprox = np.append(chebyClen(value[0][:25], n//2, -1, 0, x_neg), chebyClen(value[0][25:], n//2, 0, 1, x_pos))
                errors.append(np.max(np.abs(chebyApprox - value[1])))

            else:
                chebyApprox = chebyClen(value[0], n, -1, 1, x)
                errors.append(np.max(np.abs(chebyApprox - value[1])))
            
        plt.figure(figsize=(10, 7))
        plt.title(r"Absolute Error in Estimate of " + r"${}(x)$".format(name))
        plt.semilogy(N, errors)
        plt.xlabel(r"$n\rightarrow$")
        plt.ylabel(r"$Error\rightarrow$")
        plt.savefig("./imgs/" + name.replace("\\", "") + '_cheby.png')
        plt.close()

    
    coeffs_fourier = {
                "f_coeff": [np.abs(fourierFit(f, 50)), f(x)],
                "g_coeff": [np.abs(fourierFit(g.g, 50)), g.g(x)],
                "h_coeff": [np.abs(fourierFit(h.h, 50)), h.h(x)],
                "u_coeff": [np.abs(fourierFit(u, 50)), u(x)],
                "u\_break_coeff": [np.abs(fourierFit(u, 50, True)), u(x)],
                "v_coeff": [np.abs(fourierFit(v, 50)), v(x)]    
    }   

    n = len(coeffs_fourier["f_coeff"][0])
    plt.title("Coefficient magnitude for f")
    plt.semilogy(range(-(n//2), n//2+1), coeffs_fourier["f_coeff"][0])
    plt.xlabel(r"$n\rightarrow$")
    plt.ylabel(r"$Mag\rightarrow$")
    plt.savefig('./imgs/coeffF.png')
    plt.close()

    n = len(coeffs_fourier["u_coeff"][0])
    plt.title("Coefficient magnitude for u")
    plt.semilogy(range(-(n//2), n//2+1), coeffs_fourier["u_coeff"][0])
    plt.xlabel(r"$n\rightarrow$")
    plt.ylabel(r"$Mag\rightarrow$")
    plt.savefig('./imgs/coeffU.png')
    plt.close()

    n = len(coeffs_fourier["u\_break_coeff"][0])
    plt.title("Coefficient magnitude for u")
    plt.semilogy(range(-(n//2), n//2), coeffs_fourier["u\_break_coeff"][0])
    plt.xlabel(r"$n\rightarrow$")
    plt.ylabel(r"$Mag\rightarrow$")
    plt.savefig('./imgs/coeffUbreak.png')
    plt.close()

    n = len(coeffs_fourier["v_coeff"][0])
    plt.title("Coefficient magnitude for v")
    plt.semilogy(range(-(n//2), n//2+1), coeffs_fourier["v_coeff"][0])
    plt.xlabel(r"$n\rightarrow$")
    plt.ylabel(r"$Mag\rightarrow$")
    plt.savefig('./imgs/coeffV.png')
    plt.close()

    n = len(coeffs_fourier["g_coeff"][0])
    plt.title(r"Coefficient magnitude for g $(\delta=1)$")
    plt.semilogy(range(-(n//2), n//2+1), coeffs_fourier["g_coeff"][0])
    plt.xlabel(r"$n\rightarrow$")
    plt.ylabel(r"$Mag\rightarrow$")
    plt.savefig('./imgs/coeffG1.png')
    plt.close()

    n = len(coeffs_fourier["h_coeff"][0])
    plt.title(r"Coefficient magnitude for H $(\delta=1)$")
    plt.semilogy(range(-(n//2), n//2+1), coeffs_fourier["h_coeff"][0])
    plt.xlabel(r"$n\rightarrow$")
    plt.ylabel(r"$Mag\rightarrow$")
    plt.savefig('./imgs/coeffH1.png')
    plt.close()

    g.change_delta(0.3);h.change_delta(0.3)
    coeffs_fourier['g_coeff'][0] = fourierFit(g.g, 50)
    coeffs_fourier['h_coeff'][0] = fourierFit(h.h, 50)

    n = len(coeffs_fourier["g_coeff"][0])
    plt.title(r"Coefficient magnitude for g $(\delta=0.3)$")
    plt.semilogy(range(-(n//2), n//2+1), coeffs_fourier["g_coeff"][0])
    plt.xlabel(r"$n\rightarrow$")
    plt.ylabel(r"$Mag\rightarrow$")
    plt.savefig('./imgs/coeffG0.3.png')
    plt.close()

    n = len(coeffs_fourier["h_coeff"][0])
    plt.title(r"Coefficient magnitude for H $(\delta=0.3)$")
    plt.semilogy(range(-(n//2), n//2+1), coeffs_fourier["h_coeff"][0])
    plt.xlabel(r"$n\rightarrow$")
    plt.ylabel(r"$Mag\rightarrow$")
    plt.savefig('./imgs/coeffH0.3.png')
    plt.close()

    g.change_delta(3);h.change_delta(3)
    coeffs_fourier['g_coeff'][0] = fourierFit(g.g, 50)
    coeffs_fourier['h_coeff'][0] = fourierFit(h.h, 50)
    coeffs_fourier['g_coeff'][1] = g.g(x)
    coeffs_fourier['h_coeff'][1] = h.h(x)

    n = len(coeffs_fourier["g_coeff"][0])
    plt.title(r"Coefficient magnitude for g $(\delta=3)$")
    plt.semilogy(range(-(n//2), n//2+1), coeffs_fourier["g_coeff"][0])
    plt.xlabel(r"$n\rightarrow$")
    plt.ylabel(r"$Mag\rightarrow$")
    plt.savefig('./imgs/coeffG3.png')
    plt.close()

    len(coeffs_fourier["h_coeff"][0])
    plt.title(r"Coefficient magnitude for H $(\delta=3)$")
    plt.semilogy(range(-(n//2), n//2+1), coeffs_fourier["h_coeff"][0])
    plt.xlabel(r"$n\rightarrow$")
    plt.ylabel(r"$Mag\rightarrow$")
    plt.savefig('./imgs/coeffH3.png')
    plt.close()

    
    for key, value in coeffs_fourier.items():
        name = key.split("_coeff")[0]
        error = np.abs(fourierClen(value[0], x) - value[1])

        plt.figure(figsize=(10, 7))
        plt.title("Error in Fourier Estimate of " + r"{}(x)".format(name))
        plt.semilogy(x, error)
        plt.xlabel(r'$x\rightarrow$')
        plt.ylabel(r'$Error\rightarrow$')
        plt.savefig("./imgs/" + name.replace("\\", "") + "_error.png")


if __name__ == "__main__":
    #question1()
    #question2()
    question3()