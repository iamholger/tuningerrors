#!/usr/bin/env python
# -*- python -*-
import numpy as np


def plotEigenTuneHist(jd, ETS):
    import pyrapp
    R = [pyrapp.Rapp(r) for r in jd["RAPP"]]


    TUNED = [r(jd["TUNEPARS"]) for r in R ]
    ETD=[]
    for ET in ETS:
        ETD.append([[r(e) for r in R ]    for e in ET])

    phibest=jd["WINNERPHI2"]
    phibestpars = jd["WINNERPARS"]

    W = [r(phibestpars) for r in R]

    ERR = np.sqrt(1./np.array(jd["TUNEINVDATACOV"]).reshape((jd["NBINS"],jd["NBINS"])).diagonal())
    YV  = jd["YVALS"]

    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')
    plt.figure(figsize=(8,8))
    #
    X = np.array(range(len(YV)))+0.5
    plt.errorbar(X, YV, yerr=ERR, marker="o", linestyle="none", color="k")
    plt.yscale("log")
    plt.xlabel("# Bin")
    plt.ylabel("Entries")
    plt.plot(X, TUNED, "r-", label="Central tune (%f)"%jd["TUNECHI2"])

    etcols=["b", "m"]
    for num, ET in enumerate(ETD):
        plt.plot(X, ET[0], "%s-"%etcols[num], label="ET%i+"%(num+1))
        plt.plot(X, ET[1], "%s--"%etcols[num], label="ET%i-"%(num+1))

    plt.plot(X, W, "g--", label="Phi best(%f)"%phibest)
    plt.legend()
    plt.ylim((100, 580))
    #
    plt.savefig("toy-et-dist-%i-%s.pdf"%(jd["NBINS"], jd["CORRMODE"]))

def eigenDecomposition(mat):
    """
    Given a symmetric, real NxN matrix, M, an eigen decomposition is always
    possible, such that M can be written as M = T_transp * S * T (orthogonal
    transformation) with T_transp * T = 1_N and S being a diagonal matrix with
    the eigenvalues of M on the diagonal.

    Returns
    -------
    T_trans : numpy.matrix
    S : numpy.ndarray
        The real eigen values.
    T : numpy.matrix
    """
    from scipy import linalg
    from numpy import matrix
    import numpy

    A = matrix(mat)
    S, T_trans = linalg.eig(A)
    if numpy.iscomplex(S).any():
        raise ValueError("Given matrix `mat` has complex eigenvalues!")

    return matrix(T_trans), S.real, matrix(T_trans).transpose()

def mkPlots(jd, target):
    plotPhi2(jd["VPHI2"], jd["TUNECHI2"], jd["NDF"], "toy-phi2s-ndf-%i-%s.pdf"%(jd["NBINS"], jd["CORRMODE"]), target=target)
    plotChi2(jd["VCHI2NAIVE"], jd["VCHI2FULL"], jd["NBINS"], "toy-chi2s-%i-%s.pdf"%(jd["NBINS"], jd["CORRMODE"]))

def getEigentunes(jd, target):
    nparams = jd["NBINS"] - jd["NDF"]
    Cinv_tune=np.array(jd["TUNEINVPARAMCOV"]).reshape((nparams, nparams))
    Cinv_data=np.array(jd["TUNEINVDATACOV"]).reshape((jd["NBINS"], jd["NBINS"]))
    T_fwd, S, T_back = eigenDecomposition(Cinv_tune)
    goftarget=np.percentile(jd["VPHI2"], target)
    import pyrapp
    RAPP = [pyrapp.Rapp(r) for r in jd["RAPP"]]
    ETS, AX = mkEigenTunes(T_fwd, np.array(jd["TUNEPARS"]),  lambda x:chi2wCov(jd["YVALS"], RAPP, Cinv_data, x), goftarget, prefix="toy-ets-%i-%s"%(jd["NBINS"], jd["CORRMODE"]))
    return ETS, AX

def plotEigenTunesDist(xedges, xmids, yvals, yerrs, corr, MIN, MAX, outfname="toy-defn.pdf"):
    ## Visualise the toy model
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')
    plt.figure(figsize=(14,5))
    #
    plt.subplot(121)
    plt.fill_between(xmids, MIN, MAX)
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
    plt.savefig(outfname)
    #plt.show()

def plotPhi2(v_phi2, min_chi2, ndf, f_out, target):
    import scipy.stats as st
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')
    plt.figure(figsize=(8,8))
    goftarget=np.percentile(v_phi2, target)
    plt.hist(v_phi2, np.linspace(min(v_phi2), max(v_phi2)), alpha=0.6, density=True, label="Bootstrapped $\phi^2$")
    _x = np.linspace(0, max(v_phi2), 1000)
    plt.plot( _x, st.chi2.pdf(_x, ndf), label=r'$\chi^2(N_{df})=%i$' % (ndf))
    plt.axvline(min_chi2, color="b", label="Tuning")
    plt.axvline(goftarget, color="k", label="%.2f-th percentile of $\phi^2$"%target)
    plt.xlabel("$\phi^2$")
    plt.axvline(ndf, label="$N_{df}=%i$"%(ndf))
    plt.legend()
    plt.savefig(f_out)


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
    plt.savefig(f_out)


def plotEigentunes(center, ets, fname):
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')
    plt.figure(figsize=(8,8))
    plt.plot(center[0], center[1], "ko", label="Center")

    for et in ets:
        plt.plot( [ et[0][0], et[1][0] ], [ et[0][1], et[1][1] ], "ro-" )

    plt.axis('equal')
    plt.xlim((3,8))
    plt.ylim((5,15))
    plt.xlabel("$p_1$")
    plt.ylabel("$p_2$")
    plt.savefig(fname)

def plotEigenDirection(X, Y, target, xlabel="p", fname="eigendir.pdf"):
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')
    plt.clf()
    plt.figure(figsize=(8,8))
    plt.plot(X,Y)
    plt.axhline(target)
    plt.xlabel(xlabel)
    plt.savefig(fname)


def mkEigenTunes(T_trans, point,  GOFdef, target, prefix):
    """
    COV   ... real symmetric covariance matrix
    point ... could be any point in the true parameter space but should be
              the minimisation result i.e. the center of COV
    """
    BASE = np.eye(point.shape[0])


    # Construct all base vectors (in rotated system) with pos and neg directions
    EVS = [(T_trans*np.matrix(b).T, T_trans*np.matrix(-b).T) for b in BASE]

    # Get the eigentunes
    retv = []
    retx = []
    rv = np.matrix(point).transpose()
    ETSolve(rv, np.array(EVS[1][0]), T_trans, GOFdef, target)
    for num, ev in enumerate(EVS):
        px, pv = ETSolve(rv, np.array(ev[0]), T_trans, GOFdef, target)
        mx, mv = ETSolve(rv, np.array(ev[1]), T_trans, GOFdef, target)
        retv.append( (pv, mv))
        retx.append( (px, mx))

        XX=np.linspace(-mx, px, 100)
        YY=ETVals(XX, rv, np.array(ev[0]), T_trans, GOFdef)
        plotEigenDirection(XX,YY, target, xlabel="$E_%i$"%(num+1), fname="Eigendir_%s_%i.pdf"%(prefix, num))

    return retv, retx

def ETSolve(center, direction_t, TRAFO, GOFdef, target):
    # exec GOFdef in globals() # Note globals!

    def getVal(a):
        temp = center +  a*direction_t
        locval = GOFdef(temp.ravel().tolist()[0]) - target
        return locval

    def getP(a):
        temp = center +  a*direction_t
        return temp

    from scipy.optimize import fsolve
    x = fsolve(getVal,1) # TODO maybe this requires multi-start
    # from IPython import embed
    # embed()
    # import sys
    # sys.exit(1)
    return x, np.array(getP(x)).ravel()

def ETVals(X, center, direction_t, TRAFO, GOFdef):
    # exec GOFdef in globals() # Note globals!

    def getVal(a):
        temp = center +  a*direction_t
        return GOFdef(temp.ravel().tolist()[0])

    return [getVal(x) for x in X]

def plotToyDef(xedges, xmids, yvals, yerrs, corr, MIN, MAX, outfname="toy-defn.pdf"):
    ## Visualise the toy model
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.style.use('ggplot')
    plt.figure(figsize=(14,5))
    #
    plt.subplot(121)
    plt.fill_between(xmids, MIN, MAX)
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
    plt.savefig(outfname)
    #plt.show()

def mkData(nbins):
    ## Define the toy model
    xedges = np.linspace(0,10,nbins+1)
    xmids = np.convolve(xedges, [0.5,0.5], mode="valid")
    yvals = 5 * 1e4 * (10 + xmids)**-2
    yerrs = np.sqrt(yvals)

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
    # I = [r(x)[0] for r in R]
    I = [r(x) for r in R]
    diff = DV - np.array(I)
    # v= np.dot(np.dot(diff, InvCov), diff)
    # from IPython import embed
    # embed()
    # import sys
    # sys.exit(1)
    return sum(diff*InvCov.diagonal()*(diff))
    # return v

if __name__ == "__main__":

    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-q", "--quiet", dest="QUIET", action="store_true", default=False, help="Turn off messages and plotting")
    op.add_option("-p", "--plottoy", dest="PLOTTOY", action="store_true", default=False, help="Plot the toy definition (default: %default)")
    op.add_option("-n", dest="NSAMPLES", type=int, default=1000, help="Number of samples (default: %default)")
    op.add_option("-b", dest="NBINS", type=int, default=20, help="Number of bins (default: %default)")
    op.add_option("-s", dest="NSIGNAL", type=int, default=20, help="Number of signals to sample for parameterisation (default: %default)")
    op.add_option("-c", dest="CORRMODE", default="sane", help="Correlation mode --- none | sane | mad (default: %default)")
    op.add_option("-o", dest="OUT", default="phi2.json", help="Output file for stats (default: %default)")
    op.add_option("-t", dest="TARGET", default=95, type=float, help="G.o.F target percentile (default: %default)")
    op.add_option("--seed", dest="SEED", type=int, default=12345, help="Random seed (default: %default)")
    opts, args = op.parse_args()

    if len(args)==1:
        import json
        with open(args[0]) as f:
            jd = json.load(f)
        mkPlots(jd, opts.TARGET)

        ETS, AX = getEigentunes(jd, target=opts.TARGET)
        # This plots the et axes
        plotEigentunes(jd["TUNEPARS"], ETS, "toy-et-def-%i-%s.pdf"%(jd["NBINS"], jd["CORRMODE"]))

        # Plot the Eigentunes, central tunes and data histos
        plotEigenTuneHist(jd, ETS)

        import sys
        sys.exit(0)

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

    # Distribute the bins (objects or whatever you want to call them) amongst the available ranks
    if rank==0:
        np.random.seed(9999)
        xedges, xmids, yvals, yerrs = mkData(opts.NBINS)
        corr = mkCorr(opts.NBINS, opts.CORRMODE)

        C = mkCov(yerrs, opts.CORRMODE)
        # We need to occasionally deal with 
        #      ValueError: the input matrix must be positive semidefinite
        # https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning
        min_eig = np.min(np.real(np.linalg.eigvals(C)))
        if min_eig < 0:
            C -= 10*min_eig * np.eye(*C.shape)

        import scipy.stats as st
        # Get a once smeared sample to be used as toy data # NOTE this requires a factor two in phi2 later
        _yvals=np.ones(opts.NBINS)*-1
        while any([y<0 for y in _yvals]):
            _yvals =  st.multivariate_normal(yvals, C).rvs(size=1)
        yvals=_yvals
        yerrs = np.array([np.sqrt(y) for y in yvals])
        # Also get the sample covariance matrix based on the sample errors
        C = mkCov(yerrs, opts.CORRMODE)
        min_eig = np.min(np.real(np.linalg.eigvals(C)))
        if min_eig < 0:
            C -= 10*min_eig * np.eye(*C.shape)

        ranges = [(3.0, 8.0), (5, 15)]#, (-2.2, -1.9)]
        R, YMIN, YMAX = mkParameterisations(xmids, ranges, opts.NSIGNAL, order=(2,2))

        # This plots the toy definition
        if opts.PLOTTOY and not opts.QUIET: plotToyDef(xedges, xmids, yvals, yerrs, corr, YMIN, YMAX, outfname="toy-defn-%i-%s.pdf"%(opts.NBINS, opts.CORRMODE))

        # Here we draw samples using the Covariance matrix above
        np.random.seed(opts.SEED)
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

        # This is the covariance matrix without correlations but WITH that extra factor of sqrt(2) --- to be used when doing the
        # bootstrapping
        Cinvtrivial = np.diag([1./(2*e**2) for e in yerrs])
        # Cinvtrivial = np.diag([1./(np.sqrt(2)*e**2) for e in yerrs])

        allJobs=chunkIt(range(opts.NSAMPLES), size) # A list of lists of approximately the same length

    else:
        R = None
        sampledyvals = None
        allJobs = []
        Cinvtrivial = None

    # Scatter and broadcast operations
    comm.Barrier()
    rankJobs = comm.scatter(allJobs, root=0)
    R = comm.bcast(R, root=0)
    Cinvtrivial = comm.bcast(Cinvtrivial, root=0)
    sampledyvals = comm.bcast(sampledyvals, root=0)
    comm.Barrier()

    box = zip(R[0].pmin, R[0].pmax)
    center = R[0].pmin + 0.5*(R[0].pmax - R[0].pmin)

    # This is the bootstrapping
    restrivial = []
    minPhi2 = 1e11
    minP=[]
    from scipy import optimize
    for num, sv in enumerate(sampledyvals[rankJobs]):
        minres=optimize.minimize(lambda x:chi2wCov(sv, R, Cinvtrivial,x), center, bounds=box)
        restrivial.append(minres["fun"])
        if minres["fun"] < minPhi2:
            minPhi2 = minres["fun"]
            minP = minres["x"]
        if num%100 == 0 and rank==0: print("[{}] Done with {}/{}".format(rank, num, len(rankJobs)))

    comm.Barrier()

    # Collective operation --- gather all information on rank 0 for plotting etc.
    outputtrivial = comm.gather(restrivial, root=0)
    allminPhi2    = comm.gather(minPhi2, root=0)
    allminP       = comm.gather(minP, root=0)

    if rank==0:
        ALL = [item for sublist in outputtrivial for item in sublist]

        winner = allminPhi2.index(min(allminPhi2))
        winnerP = allminP[winner]

        # from IPython import embed
        # embed()

        # This is the tuning main minimisation --- note that this uses the covariance WITHOUT sqrt(2)
        Cinv_phi2 = np.diag([1./e**2 for e in yerrs])
        MIN_tune = optimize.minimize(lambda x:chi2wCov(yvals, R, Cinv_phi2, x), center, bounds=box)
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

        if not opts.QUIET: mkPlots(jd, opts.TARGET)
        import json
        with open(opts.OUT, "w") as f:
            json.dump(jd  ,f)

        if not opts.QUIET:
            ETs, AX = getEigentunes(jd, opts.TARGET)
            plotEigentunes(jd["TUNEPARS"], ETs, "toy-et-def-%i-%s.pdf"%(jd["NBINS"], jd["CORRMODE"]))
            plotEigenTuneHist(jd, ETs)

