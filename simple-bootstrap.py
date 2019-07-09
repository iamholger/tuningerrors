#! /usr/bin/env python

import numpy as np
import scipy.stats as st, scipy.optimize as opt


def mkData(nbins):
    xedges = np.linspace(0,10,nbins+1)
    xmids = np.convolve(xedges, [0.5,0.5], mode="valid")
    yvals = 5 * 1e4 * (10 + xmids)**-2
    yerrs = np.sqrt(abs(yvals))
    return xedges, xmids, yvals, yerrs

def mkModel(xmids, ps):
    yvals = ps[0] * 1e4 * (ps[1] + xmids)**-2
    return yvals


def chi2(datavals, modelvals, invcov):
    diff = datavals - modelvals
    return sum(diff*invcov.diagonal()*diff)

def chi2FromParams(datavals, xs, params, invcov):
    modelvals = mkModel(xs, params)
    return chi2(datavals, modelvals, invcov)


NBINS = 20
NITER = 10000
xedges, xmids, datavals, dataerrs = mkData(NBINS)
chi2s, chi2s_fit = [], []
for n in range(NITER):
    if (n+1) % 1000 == 0:
        print("Iteration #{}".format(n+1))
    datavals_smear = datavals + st.norm.rvs(0, dataerrs)
    dataerrs_smear = np.sqrt(abs(datavals_smear))
    invcov = np.diag(np.reciprocal(dataerrs**2)) #< or dataerrs_smear?
    chi2s.append(chi2(datavals, datavals_smear, invcov))
    optres = opt.minimize(lambda ps: chi2FromParams(datavals_smear, xmids, ps, invcov), [1.,1.])
    #optps = optres.x
    chi2s_fit.append(optres.fun)
xchi2s = np.linspace(0, 2.5*NBINS, 100)

import matplotlib.pyplot as plt
plt.savefigs = lambda name : [plt.savefig(name+ext, dpi=150) for ext in [".pdf", ".png"]]
plt.hist(chi2s, bins=xchi2s, density=True, label="Smear only", alpha=0.7)
plt.hist(chi2s_fit, bins=xchi2s, density=True, label="Smear+fit", alpha=0.7)
plt.plot(xchi2s, [st.chi2.pdf(x, NBINS) for x in xchi2s], label=r"$\chi^2(k = N_\mathrm{bin})$")
plt.xlabel(r"$\phi^2$")
plt.legend()
plt.savefigs("simplechi2")
plt.show()
