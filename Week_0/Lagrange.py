import os
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

def GenNoise():
    x = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    y = np.sin(np.pi*x)
    
    sigmas = np.array([0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25])

    writer = open('tmp.txt', 'w')
    
    for sigma in sigmas:
        noise = np.random.randn(9)*sigma
        yNoise = list(np.round(y + noise, 5))
        
        writer.write(str(yNoise)[1:-1] + ", ")    
    

def runC(name: str, flag: int):
    os.system('gcc -o ' + name + ".out " + name + '.c' + ' -lm')
    os.system('./' + name + '.out' + ' tmp.txt ' + str(flag))
    
    data = np.loadtxt('output.txt')
    return data[:, 0], data[:, 1]

def delete():
    os.system('rm tmp.txt')

def plot():
    GenNoise()
    x = np.arange(0, 2, 1e-5)
    yActual = np.sin(np.pi*x)

    inpt, outpt = runC('Lagrange', 0)

    plt.figure(figsize=(10,7))
    plt.plot(x, yActual, 'r-')
    plt.plot(inpt, outpt, 'b--', linewidth=2)
    plt.title("Actual vs Interpolated")
    plt.xlabel(r'$x \rightarrow$')
    plt.ylabel(r"$sin(\pi*x) \rightarrow$")
    plt.legend(['Actual Values', 'Interpolated Values'])
    plt.savefig('NoNoise.png')

    error = np.abs(np.sin(inpt*np.pi) - outpt)

    plt.figure(figsize=(10, 7))
    plt.semilogy(inpt, error, "b-")
    plt.title("Error of interpolation")
    plt.xlabel(r"$x \rightarrow$")
    plt.ylabel(r"$error \rightarrow$")
    plt.legend(["Error"])
    plt.savefig('NoNoiseError.png')

    sigmas = np.array([0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25])

    plt.figure(figsize=(10,7))
    plt.plot(x, yActual, 'b-')
    for i in range(len(sigmas)):
        inpt, outpt = runC('Lagrange', i+1)
        plt.plot(inpt, outpt, linestyle='--')
    
    plt.title("Actual Values vs Interpolated Values with Gaussian Noise ($\sigma$)")
    plt.xlabel(r"$x \rightarrow$")
    plt.ylabel(r"$\sin(\pi*x) \rightarrow$")
    plt.legend(['Actual'] + [f'$\sigma = ${sig}' for sig in sigmas])
    plt.savefig("Noise.png")

    errors = []

    plt.figure(figsize=(10,7))
    
    for i in range(len(sigmas)):
        inpt, outpt = runC('Lagrange', i+1)
        delta = np.sin(np.pi*inpt) - outpt
        errors.append(np.sum(np.abs(delta)))
        plt.semilogy(inpt, delta, linestyle='--')

    plt.title(r"Error in interpolation vs $\sigma$")
    plt.xlabel(r"$x \rightarrow$")
    plt.ylabel(r"$Error \rightarrow$")
    plt.legend([f'$\sigma = {sig}$' for sig in sigmas])
    plt.savefig('NoiseError.png')

    plt.figure(figsize=(10,7))
    plt.plot(sigmas, errors)
    plt.title(r'Absolute Error vs $\sigma$')
    plt.xlabel(r"$\sigma \rightarrow$")
    plt.ylabel(r"Error $\rightarrow$")
    plt.savefig("SigVsError.png")

    delete()

if __name__ == '__main__':
    plot()
