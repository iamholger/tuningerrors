import json, sys
import numpy as np

def construct(PTS, CTR=None, COV=None, plot_prefix=None, percentile=68):
    """
    Eigentune construction based on covariance matrix
    construction from ensemble of points (or manually specified from e.g. minimiser)
    and center.

    The points are epected to be a list-like array of P-dimensional points.
    """
    # No manually specified center or covariance, so we calculate from the ensemble
    if CTR is None and COV is None:
        CTR = np.mean(PTS, axis=0)
        COV = np.cov(PTS.T)


    # Eigenvalue problem to get rotation matrix, V
    S, V = np.linalg.eig(COV)

    # Apply rotation and translation
    EV = np.array([V.T@e for e in PTS-CTR])

    # Apply stretch, rotation and translation so that all points (EVK) are in unit sphere
    EVK = np.array([np.diag(0.5/np.sqrt(S))@V.T@e for e in PTS-CTR])

    # Calculate the radius corresponding to the percentile in question
    intercept = np.percentile(np.linalg.norm(EVK, axis=1), percentile)

    # Just some control plots
    good = np.where ( np.linalg.norm(EVK, axis=1) < intercept )
    bad  = np.where ( np.linalg.norm(EVK, axis=1) > intercept )

    # Eigentune coordinates in the unit sphere world
    dim=len(S)
    UET = intercept*np.append(-np.eye(dim), np.eye(dim), axis=1).reshape((2*dim,dim))

    # Note that we now need to apply the inverse transforms to get the real world coordinates

    # Stretch
    SET = np.array([np.diag(1/(0.5/np.sqrt(S)))@u for u in UET])

    # Rotate and translate
    ET = np.array([V@s + CTR for s in SET])


    if plot_prefix is not None and dim==2:
        plotScatter(PTS,         "{}_Rellipsis.pdf".format(   plot_prefix))
        plotScatter(PTS-CTR,   "{}_RellipsisOri.pdf".format(plot_prefix))
        plotScatter(EV,        "{}_RellipsisV.pdf".format(  plot_prefix))
        plotScatter(EVK,       "{}_RellipsisVK.pdf".format( plot_prefix))
        plotScatter(PTS[good], "{}_Rgood.pdf".format(       plot_prefix))
        plotScatter(PTS[bad],  "{}_Rbad.pdf".format(        plot_prefix))
        plotScatter(ET,        "{}_eigentunes.pdf".format(  plot_prefix))
        from matplotlib import pyplot as plt
        import matplotlib as mpl
        plt.clf()
        plt.style.use('ggplot')
        fig,ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.scatter(PTS[:,0],  PTS[:,1],  c="b", marker=".")
        ax.scatter(ET[:,0], ET[:,1], c="r", marker="x", s=100)
        ax.scatter(CTR[0], CTR[1], c="magenta", marker="*", s=100)
        plt.savefig("{}_summary.pdf".format(plot_prefix))

    return ET, CTR


assert(sys.argv[1]!=sys.argv[2])

with open(sys.argv[1]) as f:
    data = json.load(f)

D = np.array(data["DATA"])
P = D[:,range(D.shape[1]-1)]
Ptune = P[0]
PHI=D[:,-1]


try:
    PNAMES=data["PARAMS"]
except:
    PNAMES = ["PAR{}".format(i) for i in range(len(Ptune))]

TARGET=66
PLOTPREFIX=sys.argv[2]
ET, CTR = construct(P[1:], plot_prefix=PLOTPREFIX, percentile=float(TARGET))

print(ET)
for i, px in enumerate(PNAMES):
    for j, py in enumerate(PNAMES):
        if i!= j:
            print(px, py)
print(PNAMES)

from matplotlib import pyplot as plt
import matplotlib as mpl
PTS=P
for i, px in enumerate(PNAMES):
    for j, py in enumerate(PNAMES):
        if i!= j:
            plt.clf()
            plt.style.use('ggplot')
            fig, ax = plt.subplots(1)
            ax.set_aspect('equal')
            ax.scatter(PTS[:,i],  PTS[:,j],  c=PHI, marker=".", s=20,alpha=0.2)
            # ax.scatter(PTS[:,i],  PTS[:,j],  c="b", marker=".", s=20,alpha=0.2)
            ax.scatter(ET[:,i], ET[:,j], c="r", marker="x", s=50, alpha=0.8)
            ax.scatter(CTR[i], CTR[j], c="cyan", marker="o", s=100, alpha=0.5)
            ax.scatter(Ptune[i], Ptune[j], c="gold", marker="*", s=100, alpha=0.5)
            plt.xlabel(px)
            plt.ylabel(py)
            plt.savefig("{}_ellipsis_{}_{}.pdf".format(PLOTPREFIX, i, j))


