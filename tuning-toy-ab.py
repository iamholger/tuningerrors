#!/usr/bin/env python
# -*- python -*-
import numpy as np

from tuningerrors import ellipsis

from matplotlib import pyplot as plt
def savefigs(name, dpi=150):
    for ext in [".pdf", ".png"]:
        #print(name+ext, dpi)
        plt.savefig(name+ext, dpi=dpi)
plt.savefigs = savefigs


def plotEigenTuneHist2(jd, xmids, central, ETs, truth, plot_prefix=""):

    TUNED = np.array(mkSignal2d(central, xmids))
    ETD = [
            [np.array(mkSignal2d(ETs[0], xmids)), np.array(mkSignal2d(ETs[1], xmids))],
            [np.array(mkSignal2d(ETs[2], xmids)), np.array(mkSignal2d(ETs[3], xmids))]
            ]

    # from IPython import embed
    # embed()
    dfp = []
    for num, y in enumerate(TUNED):
        a = max(map(abs, [ETD[0][0][num]-y, ETD[0][1][num]-y]))
        b = max(map(abs, [ETD[1][0][num]-y, ETD[1][1][num]-y]))
        dfp.append(np.sqrt(a**2+b**2))

    dfm = []
    for num, y in enumerate(TUNED):
        a = max(map(abs, [y-ETD[0][0][num], y-ETD[0][1][num]]))
        b = max(map(abs, [y-ETD[1][0][num], y-ETD[1][1][num]]))
        dfm.append(np.sqrt(a**2+b**2))

    DFP=TUNED+dfp
    DFM=TUNED-dfm

    ERR = np.sqrt(1./np.array(jd["TUNEINVDATACOV"]).reshape((jd["NBINS"],jd["NBINS"])).diagonal())
    YV  = jd["YVALS"]

    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')
    plt.figure(figsize=(10,8))
    #
    X = np.array(range(len(YV)))+0.5
    plt.fill_between(X, DFP, DFM, alpha=0.5, color="red")
    plt.plot(X,truth, label="Truth", linewidth=3, color="black")
    plt.yscale("log")
    plt.xlabel("# Bin")
    plt.ylabel("Entries")

    etcols=["b", "m"]
    for num, ET in enumerate(ETD):
        plt.plot(X, ET[0], "%s-"%etcols[num], linewidth=3, label="ET%i+"%(num+1))
        plt.plot(X, ET[1], "%s--"%etcols[num], linewidth=3, label="ET%i-"%(num+1))

    plt.errorbar(X, YV, yerr=ERR, marker="o", linestyle="none", color="k", label="'Data'")
    plt.plot(X, TUNED, "r-", linewidth=3, label="Central tune ($\chi^2=%.1f$)"%jd["TUNECHI2"])
    plt.legend()
    plt.ylim((100, 580))
    #
    plt.savefigs("result-%s-%i-%s" % (plot_prefix, jd["NBINS"], jd["CORRMODE"]))
    plt.close()


def plotPhi2(v_phi2, min_chi2, ndf, f_out, target):
    import scipy.stats as st
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')
    plt.figure(figsize=(8,8))
    goftarget=np.percentile(v_phi2, target)
    goftarget68=np.percentile(v_phi2, 68.8)
    plt.hist(v_phi2, np.linspace(min(v_phi2), max(v_phi2)), alpha=0.6, density=True, label="Bootstrapped $\phi^2$")
    _x = np.linspace(0, max(v_phi2), 1000)
    plt.plot( _x, st.chi2.pdf(_x, ndf), label=r'$\chi^2(N_{df})=%i$' % (ndf))
    plt.axvline(min_chi2, color="b", label="Tuning")
    plt.axvline(goftarget, color="k", label="Target %.2f-th percentile of $\phi^2$"%target)
    plt.axvline(goftarget68, color="m", label="68.8-th percentile of $\phi^2$")
    plt.xlabel("$\phi^2$")
    plt.axvline(ndf, label="$N_{df}=%i$"%(ndf))
    plt.legend()
    plt.savefigs(f_out)
    plt.close()


def plotChi2(v_chi2, v_chi2_full, ndf, f_out):
    import scipy.stats as st
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')
    plt.figure(figsize=(8,8))
    plt.hist(v_chi2, np.linspace(min(v_chi2), max(v_chi2)), alpha=0.6, density=True, label="Bootstrapped $\chi^2$ w/o correlation")
    plt.hist(v_chi2_full, np.linspace(min(v_chi2_full), max(v_chi2_full)), alpha=0.6, density=True, label="Bootstrapped $\chi^2$ with correlation")
    _x = np.linspace(0, max(v_chi2), 1000)
    plt.plot( _x, st.chi2.pdf(_x, ndf), label=r'$\chi^2(N_{df})=%i$' % (ndf))
    plt.axvline(ndf, label="$N_{df}=%i$"%(ndf))
    plt.xlabel("$\chi^2$")
    plt.legend()
    plt.savefigs(f_out)
    plt.close()



def plotToyDef(xedges, xmids, yvals, yerrs, corr, MIN, MAX, outfname="toy-defn"):
    ## Visualise the toy model
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')
    plt.figure(figsize=(14,5))
    #
    plt.subplot(121)
    plt.fill_between(xmids, MIN, MAX, color="pink")
    plt.errorbar(xmids, yvals, xerr=xedges[1:]-xmids, yerr=yerrs, marker="o", linestyle="none", color="k")
    plt.yscale("log")
    plt.xlabel("Observable")
    plt.ylabel("Entries")
    #
    plt.subplot(122)
    plt.imshow(corr, interpolation="none", norm=mpl.colors.Normalize(-1,1), cmap="RdBu_r")
    #plt.matshow(cov)
    plt.colorbar()
    #
    plt.savefigs(outfname)
    #plt.show()
    plt.close()


def mkData(nbins):
    ## Define the toy model
    xedges = np.linspace(0,10,nbins+1)
    xmids = np.convolve(xedges, [0.5,0.5], mode="valid")
    yvals = 5 * 1e4 * (10 + xmids)**-2
    yerrs = np.sqrt(abs(yvals))
    return xedges, xmids, yvals, yerrs

def mkSignal2d(p, xmids):
    yvals = p[0] * 1e4 * (p[1] + xmids)**-2
    return yvals

def mkSignal(p, xmids):
    yvals = p[0] * 1e4 * (p[1] + xmids)**p[2]
    return yvals


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


def mkParameterisations(xmids, ranges, nsignal, order=(3,0)):
    import pyrapp
    S=pyrapp.sampling.NDSampler(ranges)
    X = [S() for _ in range(nsignal)]
    Y = np.array([mkSignal2d(x, xmids) for x in X])
    MIN = Y.min(axis=0)
    MAX = Y.max(axis=0)
    import pyrapp
    return [pyrapp.Rapp(X, y, order=order, strategy=1) for y in Y.T], MIN, MAX


# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length/2136090?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    # Fix size, sometimes there is spillover
    # TODO: replace with while if problem persists
    if len(out)>num:
        out[-2].extend(out[-1])
        out=out[0:-1]

    if len(out)!=num:
        raise Exception("something went wrong in chunkIt, the target size differs from the actual size")

    return out


def chi2wCov(DV, R, InvCov, x):
    import pyrapp
    if type(R[0]) == pyrapp.rapp.Rapp:
        I = [r(x) for r in R]
    else:
        I = mkSignal2d(x, R) # R is xmids here
    diff = DV - np.array(I)
    return sum(diff*InvCov.diagonal()*(diff))



if __name__ == "__main__":

    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-q", "--quiet", dest="QUIET", action="store_true", default=False, help="Turn off messages and plotting")
    op.add_option("-p", "--plottoy", dest="PLOTTOY", action="store_true", default=False, help="Plot the toy definition (default: %default)")
    op.add_option("--plotprefix", dest="PLOTPREFIX", default="toy6", help="Plot prefix (default: %default)")
    op.add_option("-n", dest="NSAMPLES", type=int, default=1000, help="Number of samples (default: %default)")
    op.add_option("-b", dest="NBINS", type=int, default=20, help="Number of bins (default: %default)")
    op.add_option("-s", dest="NSIGNAL", type=int, default=100, help="Number of signals to sample for parameterisation (default: %default)")
    op.add_option("-c", dest="CORRMODE", default="sane", help="Correlation mode --- none | sane | mad (default: %default)")
    op.add_option("-o", dest="OUT", default="phi2.json", help="Output file for stats (default: %default)")
    op.add_option("-t", dest="TARGET", default=95, type=float, help="G.o.F target percentile (default: %default)")
    op.add_option("--seed", dest="SEED", type=int, default=12345, help="Random seed (default: %default)")
    opts, args = op.parse_args()

    rank=0
    size=1
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    except Exception as e:
        print("Exception when trying to import mpi4py:", e)
        pass

    if rank == 0:
        print("Start")

    # Distribute the bins (objects or whatever you want to call them) amongst the available ranks
    if rank==0:
        import time
        np.random.seed(opts.SEED)
        ts=time.time()
        xedges, xmids, yvals, yerrs = mkData(opts.NBINS)
        corr = mkCorr(opts.NBINS, opts.CORRMODE)

        # This plots the toy definition
        if opts.PLOTTOY and not opts.QUIET:
            plotToyDef(xedges, xmids, yvals, yerrs, corr, np.min(yvals-yerrs), np.min(yvals+yerrs),
                       outfname="toy-defn-%i-%s" % (opts.NBINS, opts.CORRMODE))

        truth = yvals
        trutherrs = yerrs
        Ctrue = mkCov(yerrs, opts.CORRMODE)
        # We need to occasionally deal with
        #      ValueError: the input matrix must be positive semidefinite
        # https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning
        # min_eig = np.min(np.real(np.linalg.eigvals(Ctrue)))
        # if min_eig < 0:
        #     Ctrue -= 10*min_eig * np.eye(*C.shape)

        import scipy.stats as st
        # Get a once-smeared sample to be used as toy data # NOTE this requires a factor two in phi2 later
        #_yvals=np.ones(opts.NBINS)*-1
        #while any([y<0 for y in _yvals]):
        yvals = st.multivariate_normal(yvals, Ctrue).rvs(size=1)
        yerrs = np.array([np.sqrt(abs(y)) for y in yvals])
        # Also get the sample covariance matrix based on the sample errors
        C = mkCov(yerrs, opts.CORRMODE)
        min_eig = np.min(np.real(np.linalg.eigvals(C)))
        if min_eig < 0:
            C -= 10*min_eig * np.eye(*C.shape)
        #print(C)
        C = Ctrue

        te = time.time()
        print("Data generation took {:2.2f} seconds".format(te-ts))

        ts=time.time()
        ranges = [(3.0, 8.0), (5, 15)]#, (-2.2, -1.9)]
        R, YMIN, YMAX = mkParameterisations(xmids, ranges, opts.NSIGNAL, order=(2,2))
        te = time.time()
        print("Parameterisation took {:2.2f} seconds".format(te-ts))

        ts=time.time()
        # Here we draw samples using the Covariance matrix above
        mn = st.multivariate_normal(yvals, C)
        sampledyvals = mn.rvs(size=opts.NSAMPLES)

        # This is the covariance matrix without correlations and WITHOUT the additional factor of sqrt(2)
        Cinv =  np.linalg.inv(C)
        Cinvchi2 = np.diag([1./(e**2) for e in yerrs])

        def chi2(datavals, modelvals, ivariance):
            deltas = np.atleast_2d(datavals - modelvals)
            c2 = deltas.dot(ivariance).dot(deltas.T)
            return c2[0,0]

        c2s_naive = np.empty([opts.NSAMPLES,1])
        for i in range(opts.NSAMPLES):
            c2s_naive[i] = chi2(yvals, sampledyvals[i], Cinvchi2)

        c2s_full = np.empty([opts.NSAMPLES,1])
        for i in range(opts.NSAMPLES):
            c2s_full[i] = chi2(yvals, sampledyvals[i], Cinv)

        # This is the covariance matrix without correlations but WITH that extra factor of sqrt(2)
        # --- to be used when doing the bootstrapping
        Cinvtrivial = np.diag([1./(2*e**2) for e in yerrs])
        te = time.time()
        print("Bootstrapping data took {:2.2f} seconds".format(te-ts))

        allJobs=chunkIt(range(opts.NSAMPLES), size) # A list of lists of approximately the same length

    else:
        R = None
        sampledyvals = None
        allJobs = []
        Cinvtrivial = None
        xmids=None

    # Scatter and broadcast operations
    comm.Barrier()
    rankJobs = comm.scatter(allJobs, root=0)
    R = comm.bcast(R, root=0)
    Cinvtrivial = comm.bcast(Cinvtrivial, root=0)
    sampledyvals = comm.bcast(sampledyvals, root=0)
    xmids = comm.bcast(xmids, root=0)
    comm.Barrier()

    box = [i for i in zip(R[0].pmin, R[0].pmax)]
    center = R[0].pmin + 0.5*(R[0].pmax - R[0].pmin)

    # This is the bootstrapping
    restrivial = []
    minPhi2 = 1e11
    allPhi2 = []
    minP=[]
    allP=[]
    from scipy import optimize
    for num, sv in enumerate(sampledyvals[rankJobs]):
        # TODO add multistart or similar
        minres = optimize.minimize(lambda x:chi2wCov(sv, xmids, Cinvtrivial, x), center) #, bounds=box)
        restrivial.append(minres["fun"])
        allP.append(minres["x"])
        allPhi2.append(minres["fun"])
        if minres["fun"] < minPhi2:
            minPhi2 = minres["fun"]
            minP = minres["x"]
        if num%100 == 0: #and rank==0:
            print("[{}] Done with {}/{}".format(rank, num, len(rankJobs)))

    comm.Barrier()

    # Collective operation --- gather all information on rank 0 for plotting etc.
    outputtrivial = comm.gather(restrivial, root=0)
    allminPhi2    = comm.gather(minPhi2, root=0)
    allminP       = comm.gather(minP, root=0)
    allallPhi2    = comm.gather(allPhi2, root=0)
    allallP       = comm.gather(allP, root=0)

    if rank==0:
        ALL = [item for sublist in outputtrivial for item in sublist]
        ALLP = np.array([item for sublist in allallP for item in sublist])

        # Sample covariance in param space and ET construction
        # COV_sample     = np.cov(np.array(ALLP).T)
        # COV_sample_inv = np.linalg.inv(COV_sample)
        ET, CTR = ellipsis.construct(ALLP, plot_prefix=opts.PLOTPREFIX, percentile=float(opts.TARGET))

        iwin  = np.argmin(allminPhi2)
        winnerP = allminP[iwin]
        winnerPhi2 = allminPhi2[iwin]

        import matplotlib.pyplot as plt
        plt.figure()
        plt.hist(np.array(allallPhi2).flatten(), 100, density=True, color="red")
        plt.axvline(winnerPhi2, c="black")
        xchi2 = np.linspace(0, 50, 200)
        fchi2 = st.chi2.pdf(xchi2, opts.NBINS-1)
        plt.plot(xchi2, fchi2, color="blue")
        plt.savefigs(opts.PLOTPREFIX+"_phi2")
        plt.close()

        # This is the tuning main minimisation --- note that this uses the covariance WITHOUT sqrt(2)
        Cinv_phi2 = np.diag([1./e**2 for e in yerrs])

        # Note add multistart here
        MIN_tune = optimize.minimize(lambda x:chi2wCov(yvals, xmids, Cinv_phi2, x), center, bounds=box)
        Cinv_tune=MIN_tune["hess_inv"].todense()

        jd = {
                "CORRMODE"   : opts.CORRMODE,
                "NBINS"      : opts.NBINS,
                "NDF"        : opts.NBINS-2, # TODO eventually nparams
                "TUNECHI2"   : MIN_tune["fun"],
                "TUNEPARS"   : list(MIN_tune["x"].ravel()),
                "WINNERPARS"   : list(winnerP),
                "WINNERPHI2"   : min(allminPhi2),
                "TUNEINVDATACOV" : list(Cinv_phi2.ravel()),
                "TUNEINVPARAMCOV" : list(Cinv_tune.ravel()),
                "VCHI2NAIVE" : list(c2s_naive.ravel()),
                "VCHI2FULL"  : list(c2s_full.ravel()),
                "VPHI2": ALL,
                "RAPP": [r.asDict for r in R],
                "YVALS": list(yvals)
                }


        oo = MIN_tune["x"]

        plotEigenTuneHist2(jd, xmids, oo, ET, truth, plot_prefix=opts.PLOTPREFIX)
