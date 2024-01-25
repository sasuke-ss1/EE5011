from polint import polint
import numpy as np
import matplotlib.pyplot as plt

def func1(lenx, xx, n, q):
    x = np.linspace(0, 1, lenx)
    error = []

    for i in n:
        yy, dyy, xx = polint(x, np.sin(x+x**2), xx, int(i))
        error.append(np.array([np.mean(np.abs(yy - np.sin(xx+xx**2))), np.max(np.abs(yy - np.sin(xx+xx**2)))]))
        
        print("We sampled X at {} points".format(lenx))
        print("We used {}th degree polynomial interpolation\n".format(i))
        print("Average and Max Error are {} and {} respectively.".format(error[-1][0], error[-1][1]))

        plt.figure(figsize=(10,7))
        plt.plot(xx, yy, 'ro')
        plt.plot(xx, np.sin(xx+xx**2), 'b')
        plt.title("{}th degree polynomial".format(i))
        plt.xlabel(r'x$\rightarrow$')
        plt.ylabel(r'y$\leftarrow$')
        plt.savefig('{}_{}_fit.png'.format(q, i))

        plt.figure(figsize=(10,7))
        plt.semilogy(xx, np.abs(yy-np.sin(xx+xx**2)), 'ro')
        plt.semilogy(xx, dyy, 'b')
        plt.title("Error in polynomial interpolation")
        plt.xlabel(r'x$\rightarrow$')
        plt.ylabel(r'error$\leftarrow$')
        plt.legend(['Error', 'Derivative'])
        plt.savefig('{}_{}_error.png'.format(q, i))

    return np.array(error)

def question1():
    n = np.array([4])
    lenx = 5
    xx = np.linspace(-0.5, 1.5, 200)
    func1(lenx, xx, n, 1)

def question2():
    n = np.array([4])
    lenx = 30
    xx = np.linspace(-0.5, 1.5, 200)
    func1(lenx, xx, n, 2)

def question34():
    n = np.arange(3, 21, 1)
    lenx = 30
    xx = np.linspace(-0.5, 1.5, 200)
    error = func1(lenx, xx, n, 34)
    
    plt.figure(figsize=(10, 7))
    plt.semilogy(n, error[:, 0], 'r')
    plt.semilogy(n, error[:, 1], 'b')
    plt.title("Error in polynomial interpolation")
    plt.xlabel(r'Degree of Polynomial$\rightarrow$')
    plt.ylabel(r'Error$\rightarrow$')
    plt.legend(["Average Error", "Maximum Error"])

    plt.savefig("AvgMax.png")

def question5():
    n = np.arange(6, 16, 1)
    x = np.arange(0.1, 0.9+0.05, 0.05)
    xx = np.linspace(0.1, 0.9, 1000)
    y = np.sin(np.pi*x)/np.sqrt(1-x**2)
    
    error = []

    for i in n:
        yy, dyy, xx = polint(x, y, xx, i)
        error.append(np.max(np.abs(yy-np.sin(xx*np.pi)/np.sqrt(1-xx**2))))
        print("We sampled X at {} points".format(len(x)))
        print("We used {}th degree polynomial interpolation\n".format(i))
        print("Max Error is {}".format(error[-1]))

        plt.figure(figsize=(10,7))
        plt.plot(xx, yy, 'ro')
        plt.plot(xx, np.sin(xx*np.pi)/np.sqrt(1-xx**2))
        plt.title("{}th degree polynomial".format(i))
        plt.xlabel(r'x$\rightarrow$')
        plt.ylabel(r'y$\leftarrow$')
        plt.savefig('5_{}_fit.png'.format(i))

        plt.figure(figsize=(10,7))
        plt.semilogy(xx, np.abs(yy-np.sin(xx*np.pi)/np.sqrt(1-xx**2)), 'ro')
        plt.semilogy(xx, dyy, 'b')
        plt.title("Error in polynomial interpolation")
        plt.xlabel(r'x$\rightarrow$')
        plt.ylabel(r'error$\leftarrow$')
        plt.legend(['Error', 'Derivative'])
        plt.savefig('5_{}_error.png'.format(i))

        
    for err in error:
        if err < 1e-5:
            print("Order of Polynomial Interpolation that produces less than 0.000001 error: {}".format(n[error.index(err)]))
            print("We get an error of {} for {}th order polynomial".format(err, n[error.index(err)]))

if __name__ == '__main__':
    question1()
    question2()
    question34()
    question5()


