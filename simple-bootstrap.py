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
    #print(datavals.shape, modelvals.shape)
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
ap.add_argument("--resample-data", dest="RESAMPLE_DATA", action="store_true", default=False)
args = ap.parse_args()


## Make perfect pseudodata and correlations
xedges, xmids, datavals, dataerrs, datacov = mkData(args.NDATABINS, args.CORRMODE)
np.random.seed(12345)

## Smear once, from the perfect distribution and with the true covariance
mn1 = st.multivariate_normal(datavals, datacov)
if args.RESAMPLE_DATA:
    datavals_smear1 = mn1.rvs(size=args.NITERS)
else:
    datavals_smear1 = np.tile(mn1.rvs(size=1), (args.NITERS,1))
#print(datavals_smear1)

## Fit nominal
chi2s, chi2s_fit, fitdata = [], [], []
invcov = np.linalg.inv(datacov) if args.CORRCHI2 else np.diag(np.reciprocal(dataerrs**2))
chi2s.append(chi2(datavals, datavals_smear1[0], invcov))
optres1 = opt.minimize(lambda ps: chi2FromParams(datavals_smear1[0], xmids, ps, invcov), [5.,10.])
chi2s_fit.append(optres1.fun)
fitdata.append([optres1.fun, optres1.x])

## Bootstrap and re-fit
# TODO: convert to depend on bootstrap chi2 procedure
#mn2 = st.multivariate_normal(datavals_smear1, datacov)
for n in range(args.NITERS):
    mn2 = st.multivariate_normal(datavals_smear1[n], datacov)
    if (n+1) % 1000 == 0:
        print("Iteration #{}".format(n+1))
    datavals_smear2 = mn2.rvs(size=1)
    #dataerrs_smear2 = np.sqrt(abs(datavals_smear2))
    #invcov = np.diag(np.reciprocal(dataerrs_smear**2))
    chi2s.append(chi2(datavals, datavals_smear2, 0.5*invcov))
    #chi2s.append(chi2(datavals_smear1, datavals_smear2, 2*0.5*invcov))

    optres2 = opt.minimize(lambda ps: chi2FromParams(datavals_smear2, xmids, ps, 0.5*invcov), [5.,10.])
    chi2s_fit.append(optres2.fun)
    fitdata.append([optres2.fun, optres2.x])

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
plt.axvline(chi2s[0], ls="--", color="darkblue", alpha=0.7)
plt.hist(chi2s[1:], bins=xchi2bins, density=True, histtype="stepfilled",
         color="aliceblue", edgecolor="darkblue", hatch="", label="Smear only", alpha=0.7)
plt.axvline(chi2s_fit[0], ls="--", color="darkred", alpha=0.7)
plt.hist(chi2s_fit[1:], bins=xchi2bins, density=True, histtype="stepfilled",
         color="mistyrose", edgecolor="darkred", hatch=r"", label="Smear+fit", alpha=0.7)
plt.plot(xchi2s, [st.chi2.pdf(x, args.NDATABINS) for x in xchi2s], color="darkblue", label=r"$\chi^2(k = N_\mathrm{bin})$")
plt.plot(xchi2s, [st.chi2.pdf(x, args.NDATABINS-2) for x in xchi2s], color="darkred", label=r"$\chi^2(k = N_\mathrm{bin}-2)$")
plt.xlabel(r"$\phi^2$")
plt.xlim(0, 2.5*args.NDATABINS)
plt.legend()
plt.savefigs("simplechi2")
plt.show()
