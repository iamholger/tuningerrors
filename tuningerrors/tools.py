import numpy as np

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


def mkSignal2d(p, xmids):
    yvals = p[0] * 1e4 * (p[1] + xmids)**-2
    return yvals

def mkSignal(p, xmids):
    yvals = p[0] * 1e4 * (p[1] + xmids)**p[2]
    return yvals

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
    plt.savefig("result-%s-%i-%s.pdf"%(plot_prefix, jd["NBINS"], jd["CORRMODE"]))


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
