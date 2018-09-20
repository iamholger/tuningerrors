# -*- python -*-
import numpy as np

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
    xedges = np.linspace(0,100,nbins+1)
    xmids = np.convolve(xedges, [0.5,0.5], mode="valid")
    yvals = 1e4 * (10 + xmids)**-2
    yerrs = np.sqrt(yvals)

    return xedges, xmids, yvals, yerrs

def mkSignal(p, xmids):
    yvals = p[0] * 1e4 * (p[1] + xmids)**p[2]
    return yvals

def mkCorr(nbins, corrmode):
    corr = np.eye(nbins)
    for i in range(nbins):
        for j in range(nbins):
            if i == j: continue
            if corrmode == "sane":
                corr[i,j] = (5*np.sqrt((i+1)*(j+1))/nbins * abs(i-j) + 1.0)**-0.8  #*  (-1)**(i+j)
            elif corrmode == "mad":
                corr[i,j] = np.sqrt((1.5 * abs(i-j) + 0.1))**-1
            else:
                print "Oops, unknown corrmode"
                import sys
                sys.exit(1)
    return corr

def mkCov(yerrs, corrmode="sane"):
    return np.atleast_2d(yerrs).T * np.atleast_2d(yerrs) * mkCorr(yerrs.shape[0], corrmode)


def mkParameterisations(xmids, ranges, nsignal, order=(3,0)):
    import professor2 as prof
    S=prof.sampling.NDSampler(ranges)
    X = [S() for _ in range(nsignal)]
    Y = np.array([mkSignal(x, xmids) for x in X])
    MIN = Y.min(axis=0)
    MAX = Y.max(axis=0)
    import pyrapp
    return [pyrapp.Rapp(X, y, order=order) for y in Y.T], MIN, MAX


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
    I = [r(x) for r in R]

    diff = DV - np.array(I)
    return np.dot(np.dot(diff, InvCov), diff)

if __name__ == "__main__":

    import optparse, os, sys
    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False, help="Turn on some debug messages")
    op.add_option("-q", "--quiet", dest="QUIET", action="store_true", default=False, help="Turn off messages")
    op.add_option("-n", dest="NSAMPLES", type=int, default=1000, help="Number of samples (default: %default)")
    op.add_option("-b", dest="NBINS", type=int, default=20, help="Number of bins (default: %default)")
    op.add_option("-s", dest="NSIGNAL", type=int, default=20, help="Number of signals to sample for parameterisation (default: %default)")
    op.add_option("-c", dest="CORRMODE", default="sane", help="Correlation mode sane | mad (default: %default)")
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

    # Distribute the bins (objects or whatever you want to call them) amongst the available ranks
    if rank==0:
        xedges, xmids, yvals, yerrs = mkData(opts.NBINS)
        corr = mkCorr(opts.NBINS, opts.CORRMODE)
        C = mkCov(yerrs, opts.CORRMODE)

        ranges = [(0.5, 2.0), (9, 11), (-2.2, -1.9)]
        R, YMIN, YMAX = mkParameterisations(xmids, ranges, opts.NSIGNAL, order=(3,2))
        plotToyDef(xedges, xmids, yvals, yerrs, corr, YMIN, YMAX, outfname="toy-defn-mpi.pdf")

        import scipy.stats as st
        mn = st.multivariate_normal(yvals, C)
        sampledyvals = mn.rvs(size=opts.NSAMPLES)


        Cinvtrivial = np.diag([1./e**2 for e in yerrs])
        Cinv =  np.linalg.inv(C)

        allJobs=chunkIt(range(opts.NSAMPLES), size) # A list of lists of approximately the same length


        def chi2(datavals, modelvals, ivariance):
            deltas = np.atleast_2d(datavals - modelvals)
            c2 = deltas.dot(ivariance).dot(deltas.T)
            return c2[0,0]

        c2s_naive = np.empty([opts.NSAMPLES,1])
        c2s_corr  = np.empty([opts.NSAMPLES,1])
        for i in range(opts.NSAMPLES):
            c2s_naive[i] = chi2(yvals, sampledyvals[i], Cinvtrivial)
            c2s_corr[i] = chi2(yvals, sampledyvals[i], Cinv)
        #print c2s_naive, c2s_corr
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        plt.style.use('ggplot')
        plt.figure(figsize=(8,8))
        chi2max = 4*opts.NBINS
        nchi2bins = np.sqrt(opts.NSAMPLES)/4.
        plt.hist(c2s_naive, np.linspace(0,chi2max,nchi2bins), alpha=0.6, label="Naive $\chi^2$")
        plt.hist(c2s_corr,  np.linspace(0,chi2max,nchi2bins), alpha=0.6, label="Correlated $\chi^2$")
        plt.xlabel("$\chi^2$")
        plt.legend()
        plt.ylim((0,5000))
        plt.axvline(opts.NBINS)
        plt.savefig("toy-chi2s-mpi.pdf")
    else:
        R = None
        sampledyvals = None
        allJobs = []
        Cinv = None
        Cinvtrivial = None

    # Scatter and broadcast operations
    comm.Barrier()
    rankJobs = comm.scatter(allJobs, root=0)
    R = comm.bcast(R, root=0)
    Cinv = comm.bcast(Cinv, root=0)
    Cinvtrivial = comm.bcast(Cinvtrivial, root=0)
    sampledyvals = comm.bcast(sampledyvals, root=0)
    comm.Barrier()

    box = zip(R[0].pmin, R[0].pmax)
    center = R[0].pmin + 0.5*(R[0].pmax - R[0].pmin)

    res, restrivial = [], []
    from scipy import optimize
    for num, sv in enumerate(sampledyvals[rankJobs]):
        res.append(       optimize.minimize(lambda x:chi2wCov(sv, R, Cinv,x),        center, bounds=box)["fun"])
        restrivial.append(optimize.minimize(lambda x:chi2wCov(sv, R, Cinvtrivial,x), center, bounds=box)["fun"])
        if num%100 == 0: print "[{}] Done with {}/{}".format(rank, num, len(rankJobs))

    comm.Barrier()

    # Collective operation --- gather the long strings from each rank
    output = comm.gather(res, root=0)
    outputtrivial = comm.gather(restrivial, root=0)

    if rank==0:
        ALL =        [item for sublist in output        for item in sublist]
        ALLtrivial = [item for sublist in outputtrivial for item in sublist]
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        plt.style.use('ggplot')
        plt.figure(figsize=(8,8))
        chi2max = 4*opts.NBINS
        nchi2bins = np.sqrt(opts.NSAMPLES)/4.
        plt.hist(ALLtrivial, np.linspace(0,chi2max,nchi2bins), alpha=0.6, label="Naive $\phi^2$")
        plt.hist(ALL,        np.linspace(0,chi2max,nchi2bins), alpha=0.6, label="Correlated $\phi^2$")
        plt.xlabel("$\phi^2$")
        plt.axvline(opts.NBINS-3)
        plt.ylim((0,5000))
        plt.legend()
        plt.savefig("toy-phi2s-mpi.pdf")

# ## Make 10-bin data samples drawn from correlated distributions
# ## (the covariance encodes various potential correlation sources:
# ## e.g. event sharing, normalization, systematics, fitting, ...)

# import numpy as np
# np.set_printoptions(precision=2, suppress=True)

# nbin = 20
# nsample = 10000
# corrmode = "mad"#"sane" #"mad"

# ## Define the toy model
# xedges = np.linspace(0,100,nbin+1)
# xmids = np.convolve(xedges, [0.5,0.5], mode="valid")
# yvals = 1e4 * (10 + xmids)**-2
# yerrs = np.sqrt(yvals)



# ranges = [(0.5, 2.0), (9, 11), (-2.2, -1.9)]

# import professor2 as prof
# S=prof.sampling.NDSampler(ranges)

# import pyrapp
# X = [S() for _ in range(100)]
# Y = np.array([mkSignal(x, xmids) for x in X])
# R = [pyrapp.Rapp(X, y, order=(3,0)) for y in Y.T]
# MIN = Y.min(axis=0)
# MAX = Y.max(axis=0)




# # print yerrs, yerrs/yvals

# ## Define correlations
# corr = np.eye(nbin)
# for i in range(nbin):
    # for j in range(nbin):
        # if i == j: continue
        # if corrmode == "sane":
            # corr[i,j] = (5*np.sqrt((i+1)*(j+1))/nbin * abs(i-j) + 1.0)**-0.8  #*  (-1)**(i+j)
        # elif corrmode == "mad":
            # corr[i,j] = np.sqrt((1.5 * abs(i-j) + 0.1))**-1
        # else:
            # print "Oops, unknown corrmode"
            # exit(1)
# #
# cov = np.atleast_2d(yerrs).T * np.atleast_2d(yerrs) * corr
# # print corr
# # print cov


# ## Visualise the toy model
# from matplotlib import pyplot as plt
# import matplotlib as mpl
# plt.style.use('ggplot')
# plt.figure(figsize=(14,5))
# #
# plt.subplot(121)
# plt.fill_between(xmids, MIN, MAX)
# plt.errorbar(xmids, yvals, xerr=xedges[1:]-xmids, yerr=yerrs, marker="o", linestyle="none", color="k")
# plt.yscale("log")
# #
# plt.subplot(122)
# plt.imshow(corr, interpolation="none", norm=mpl.colors.Normalize(-1,1), cmap="RdBu_r")
# #plt.matshow(cov)
# plt.colorbar()
# #
# plt.savefig("toy-defn.pdf")
# #plt.show()

# plt.clf()


# ## Sample from the toy model
# import scipy.stats as st
# mn = st.multivariate_normal(yvals, cov)
# sampledyvals = mn.rvs(size=nsample)
# plt.figure(figsize=(8,8))
# for i in range(min(100, nsample)):
    # plt.plot(xmids, sampledyvals[i], color="gray", alpha=0.1)
# plt.savefig("toy-samples.pdf")
# #plt.show()


# ## Visualise chi2 distribution with and without the covariance matrix
# def chi2(datavals, modelvals, variance):
    # deltas = np.atleast_2d(datavals - modelvals)
    # if len(variance.shape) == 1:
        # variance = np.diag(variance)
    # ivariance = np.linalg.inv(variance)
    # #print "VAR =\n", variance
    # #print "VAR-1 =\n", ivariance
    # c2 = deltas.dot(ivariance).dot(deltas.T)
    # #print "CHI2 =\n", c2[0,0]
    # #print
    # return c2[0,0]

# c2s_naive = np.empty([nsample,1])
# c2s_corr  = np.empty([nsample,1])
# for i in range(nsample):
    # c2s_naive[i] = chi2(yvals, sampledyvals[i], yerrs**2)
    # c2s_corr[i] = chi2(yvals, sampledyvals[i], cov)
# #print c2s_naive, c2s_corr

# plt.figure(figsize=(8,8))
# chi2max = 4*nbin
# nchi2bins = np.sqrt(nsample)/4.
# plt.hist(c2s_naive, np.linspace(0,chi2max,nchi2bins), alpha=0.6, label="Naive $\chi^2$")
# plt.hist(c2s_corr,  np.linspace(0,chi2max,nchi2bins), alpha=0.6, label="Correlated $\chi^2$")
# plt.xlabel("$\chi^2$")
# plt.legend()
# plt.savefig("toy-chi2s.pdf")


# # Minimisation
# def chi2wCov(DV, R, InvCov, x):
    # I = [r(x) for r in R]

    # diff = DV - np.array(I)
    # return np.dot(np.dot(diff, InvCov), diff)

# box = zip(R[0].pmin, R[0].pmax)
# center = R[0].pmin + 0.5*(R[0].pmax - R[0].pmin)


# from scipy import optimize

# Sinv =  np.linalg.inv(cov)


# Mcov =  optimize.minimize(lambda x:chi2wCov(yvals,R,Sinv,x), center, bounds=box)

# Sdiaginv = np.diag([1./e**2 for e in yerrs])

# Mdef = optimize.minimize(lambda x:chi2wCov(yvals,R,Sdiaginv,x), center, bounds=box)


# Rcov, Rdef = [], []
# for num, sv in enumerate(sampledyvals):
    # Rcov.append(optimize.minimize(lambda x:chi2wCov(sv,R,Sinv,    x), center, bounds=box)["fun"])
    # Rdef.append(optimize.minimize(lambda x:chi2wCov(sv,R,Sdiaginv,x), center, bounds=box)["fun"])
    # if num%10 == 0: print "Done with {}/{}".format(num+1, nsample)


# # from IPython import embed
# # embed()

# plt.figure(figsize=(8,8))
# chi2max = 4*nbin
# nchi2bins = np.sqrt(nsample)/4.
# plt.hist(Rdef, np.linspace(0,chi2max,nchi2bins), alpha=0.6, label="Naive $\phi^2$")
# plt.hist(Rcov,  np.linspace(0,chi2max,nchi2bins), alpha=0.6, label="Correlated $\phi^2$")
# plt.xlabel("$\phi^2$")
# plt.legend()
# plt.savefig("toy-phi2s.pdf")
