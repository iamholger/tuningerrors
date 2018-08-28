# -*- python -*-

## Make 10-bin data samples drawn from correlated distributions
## (the covariance encodes various potential correlation sources:
## e.g. event sharing, normalization, systematics, fitting, ...)

import numpy as np
np.set_printoptions(precision=2, suppress=True)

nbin = 20
nsample = 10000
corrmode = "sane" #"mad"

## Define the toy model
xedges = np.linspace(0,100,nbin+1)
xmids = np.convolve(xedges, [0.5,0.5], mode="valid")
yvals = 1e4 * (10 + xmids)**-2
yerrs = np.sqrt(yvals)
# print yerrs, yerrs/yvals

## Define correlations
corr = np.eye(nbin)
for i in range(nbin):
    for j in range(nbin):
        if i == j: continue
        if corrmode == "sane":
            corr[i,j] = (5*np.sqrt((i+1)*(j+1))/nbin * abs(i-j) + 1.0)**-0.8  #*  (-1)**(i+j)
        elif corrmode == "mad":
            corr[i,j] = np.sqrt((1.5 * abs(i-j) + 0.1))**-1
        else:
            print "Oops, unknown corrmode"
            exit(1)
#
cov = np.atleast_2d(yerrs).T * np.atleast_2d(yerrs) * corr
# print corr
# print cov


## Visualise the toy model
from matplotlib import pyplot as plt
import matplotlib as mpl
plt.figure(figsize=(14,5))
#
plt.subplot(121)
plt.errorbar(xmids, yvals, xerr=xedges[1:]-xmids, yerr=yerrs, marker="o", linestyle="none")
#
plt.subplot(122)
plt.imshow(corr, interpolation="none", norm=mpl.colors.Normalize(-1,1), cmap="RdBu_r")
#plt.matshow(cov)
plt.colorbar()
#
plt.savefig("toy-defn.pdf")
#plt.show()


## Sample from the toy model
import scipy.stats as st
mn = st.multivariate_normal(yvals, cov)
sampledyvals = mn.rvs(size=nsample)
plt.figure(figsize=(8,8))
for i in range(min(100, nsample)):
    plt.plot(xmids, sampledyvals[i], color="gray", alpha=0.1)
plt.savefig("toy-samples.pdf")
#plt.show()


## Visualise chi2 distribution with and without the covariance matrix
def chi2(datavals, modelvals, variance):
    deltas = np.atleast_2d(datavals - modelvals)
    if len(variance.shape) == 1:
        variance = np.diag(variance)
    ivariance = np.linalg.inv(variance)
    #print "VAR =\n", variance
    #print "VAR-1 =\n", ivariance
    c2 = deltas.dot(ivariance).dot(deltas.T)
    #print "CHI2 =\n", c2[0,0]
    #print
    return c2[0,0]

c2s_naive = np.empty([nsample,1])
c2s_corr  = np.empty([nsample,1])
for i in range(nsample):
    c2s_naive[i] = chi2(yvals, sampledyvals[i], yerrs**2)
    c2s_corr[i] = chi2(yvals, sampledyvals[i], cov)
#print c2s_naive, c2s_corr

plt.figure(figsize=(8,8))
chi2max = 4*nbin
nchi2bins = np.sqrt(nsample)/4.
plt.hist(c2s_naive, np.linspace(0,chi2max,nchi2bins), alpha=0.6, label="Naive $\chi^2$")
plt.hist(c2s_corr,  np.linspace(0,chi2max,nchi2bins), alpha=0.6, label="Correlated $\chi^2$")
plt.xlabel("$\chi^2$")
plt.legend()
plt.savefig("toy-chi2s.pdf")
