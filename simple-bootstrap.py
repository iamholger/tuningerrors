#! /usr/bin/env python
"""\
Test true and fitted phi2 values against a toy data model, with and without correlations.
"""

import numpy as np
import scipy.stats as st, scipy.optimize as opt


def mkCorr(nbins, corrmode):
    corr = np.eye(nbins)
    if corrmode == "none":
        return corr
    for i in range(nbins):
        for j in range(nbins):
            if i == j: continue
            if corrmode == "sane":
                corr[i,j] = (5*np.sqrt((i+1)*(j+1))/nbins * abs(i-j) + 1.0)**-0.8  #*  (-1)**(i+j)
            elif corrmode == "mad":
                corr[i,j] = np.sqrt((1.5 * abs(i-j) + 0.1))**-1
            else:
                print("Oops, unknown corrmode")
                import sys
                sys.exit(1)
    return corr

def mkCov(yerrs, corrmode="sane"):
    return np.atleast_2d(yerrs).T * np.atleast_2d(yerrs) * mkCorr(yerrs.shape[0], corrmode)



def mkData(nbins, corrmode="none"):
    xedges = np.linspace(0,10,nbins+1)
    xmids = np.convolve(xedges, [0.5,0.5], mode="valid")
    yvals = 5 * 1e4 * (10 + xmids)**-2
    yerrs = np.sqrt(abs(yvals))
    return xedges, xmids, yvals, yerrs, mkCov(yerrs, corrmode)

def mkModel(xmids, ps):
    yvals = ps[0] * 1e4 * (ps[1] + xmids)**-2
    return yvals


def chi2(datavals, modelvals, invcov):
    diff = datavals - modelvals
    #return sum(diff*invcov.diagonal()*diff)
    #return np.atleast_2d(diff).T @ invcov @ diff
    return diff.T @ invcov @ diff

def chi2FromParams(datavals, xs, params, invcov):
    modelvals = mkModel(xs, params)
    return chi2(datavals, modelvals, invcov)


## Parse command-line arguments
import argparse
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("-N", "--iters", dest="NITERS", type=int, default=5000)
ap.add_argument("-C", "--corrmode", dest="CORRMODE", default="sane")
ap.add_argument("-c", "--corrchi2", dest="CORRCHI2", action="store_true", default=False)
ap.add_argument("--chi2bins", dest="NCHI2BINS", type=int, default=50)
ap.add_argument("--databins", dest="NDATABINS", type=int, default=20)
args = ap.parse_args()

## Make pseudodata and correlations
xedges, xmids, datavals, dataerrs, datacov = mkData(args.NDATABINS, args.CORRMODE)
invcov = np.linalg.inv(datacov) if args.CORRCHI2 else np.diag(np.reciprocal(dataerrs**2))

## Fit nominal
chi2s, chi2s_fit, fitdata = [], [], []
optres = opt.minimize(lambda ps: chi2FromParams(datavals, xmids, ps, invcov), [1.,1.])
fitdata.append([optres.fun, optres.x])

## Bootstrap and re-fit
mn = st.multivariate_normal(datavals, datacov)
for n in range(args.NITERS):
    if (n+1) % 1000 == 0:
        print("Iteration #{}".format(n+1))
    datavals_smear = mn.rvs(size=1)
    #datavals_smear = datavals + st.norm.rvs(0, dataerrs)
    dataerrs_smear = np.sqrt(abs(datavals_smear))
    #invcov = np.diag(np.reciprocal(dataerrs_smear**2))
    # TODO: FACTOR OF 2?
    chi2s.append(chi2(datavals, datavals_smear, 2*invcov))
    optres = opt.minimize(lambda ps: chi2FromParams(datavals_smear, xmids, ps, 2*invcov), [1.,1.])
    chi2s_fit.append(optres.fun)
    fitdata.append([optres.fun, optres.x])
xchi2s = np.linspace(0, 2.5*args.NDATABINS, 100)
xchi2bins = np.linspace(0, 2.5*args.NDATABINS, args.NCHI2BINS)

## Save bootstrap points
fs = np.zeros([len(fitdata), 1])
ps = np.zeros([len(fitdata), len(fitdata[0])])
for i, fd in enumerate(fitdata):
    fs[i] = fd[0]
    ps[i,:] = fd[1]
data = np.hstack([ps, fs])
#print(data.shape)
#np.savetxt("simplebootstrap.dat", data)
import json
dd = { "DATA" : data.tolist(),
       "NBINS": args.NDATABINS,
       "PARAMS": ["p0", "p1"] }
with open("simplebootstrap.dat", "w") as out:
    json.dump(dd, out, indent=4)

## Plot
import matplotlib.pyplot as plt
plt.savefigs = lambda name : [plt.savefig(name+ext, dpi=150) for ext in [".pdf", ".png"]]
plt.hist(chi2s, bins=xchi2bins, density=True, label="Smear only", alpha=0.7)
plt.hist(chi2s_fit, bins=xchi2bins, density=True, label="Smear+fit", alpha=0.7)
plt.plot(xchi2s, [st.chi2.pdf(x, args.NDATABINS) for x in xchi2s], label=r"$\chi^2(k = N_\mathrm{bin})$")
plt.plot(xchi2s, [st.chi2.pdf(x, args.NDATABINS-2) for x in xchi2s], label=r"$\chi^2(k = N_\mathrm{bin}-2)$")
plt.xlabel(r"$\phi^2$")
plt.legend()
plt.savefigs("simplechi2")
plt.show()
