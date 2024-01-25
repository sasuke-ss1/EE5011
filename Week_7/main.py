from __future__ import division
from helper import *
from scipy.stats import chisquare, kstest
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(42)

y = BoxMuller(int(200))

plt.figure(figsize=(10, 7))
plt.hist(y, bins=5)
plt.savefig('./imgs/1.png')


# Chi-Square Test
N = [2**i for i in range(1, 11)]
bins = np.arange(-4, 5)
y = [BoxMuller(n) for n in N]
Xtab = np.array([chisquare(i) for i in y])

plt.figure(figsize=(10, 7))
plt.stem(N, Xtab[:, 0])
plt.ylabel(r'$ChiSquareValue\rightarrow$')
plt.xlabel(r'$N\rightarrow$')
plt.savefig('./imgs/2.png')

plt.figure(figsize=(10, 7))
plt.plot(N, Xtab[:, 1], label='p_values')
plt.plot(N, [0.05]*len(N), linestyle='--', label='p=0.05')
plt.legend()
plt.ylabel(r'$Pvalues\rightarrow$')
plt.xlabel(r'$N\rightarrow$')
plt.savefig('./imgs/3.png')

#Shifted 
y_offset = [i +1 for i in y]
Xtab_offest = np.array([chisquare(i) for i in y_offset])

#print(np.abs(Xtab_offest[:,0] - Xtab[:,0]))

plt.figure(figsize=(10, 7))
plt.plot(N, Xtab_offest[:, 1], label='p_values')
plt.plot(N, [0.05]*len(N), linestyle='--', label='p=0.05')
plt.legend()
plt.ylabel(r'$Pvalues\rightarrow$')
plt.xlabel(r'$N\rightarrow$')
plt.savefig('./imgs/4.png')


# KS Test
KStab = np.array([kstest(i, stats.norm.cdf) for i in y])

plt.figure(figsize=(10, 7))
plt.stem(N, KStab[:, 0])
plt.xlabel(r'$N\rightarrow$')
plt.ylabel(r'$KSValue\rightarrow$')
plt.savefig('./imgs/5.png')

plt.figure(figsize=(10, 7))
plt.plot(N, KStab[:, 1])
plt.xlabel(r'$N\rightarrow$')
plt.ylabel(r'$PValue\rightarrow$')
plt.savefig('./imgs/6.png')

y_offset = [i + 0.01 for i in y]
KStab_offset = np.array([kstest(i, stats.norm.cdf) for i in y_offset])

plt.figure(figsize=(10, 7))
plt.plot(N, KStab_offset[:, 1])
plt.xlabel(r'$N\rightarrow$')
plt.ylabel(r'$PValue\rightarrow$')
plt.savefig('./imgs/7.png')

y = Dist(900)
y20 = Dist(20)
p_chi20, p_chi = chisquare(y20)[1], chisquare(y)[1]
p_ks20, p_ks = kstest(y20, stats.norm.cdf)[1], kstest(y, stats.norm.cdf)[1]
print('p Values of Chi Square Test for 20 and 900 samples respectively: {:0.3f}, {:0.3f}'.format(p_chi20, p_chi))
print('p Values of KS Test for 20 and 900 samples respectively: {:0.3f}, {:0.3f}'.format(p_ks20, p_ks))



