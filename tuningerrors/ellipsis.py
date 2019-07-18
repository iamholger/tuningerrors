#!/usr/bin/env python

import numpy as np
import scipy.stats as st


def rdmCircle(rho=1):
    """
    Sample a point on a circle
    """
    phi = 2 * np.pi * np.random.rand()
    x = np.sqrt(rho) * np.cos(phi)
    y = np.sqrt(rho) * np.sin(phi)
    return x, y


def elliptify(x,y, a,b, theta=0.5):
    x*=a/2.
    y*=b/2.

    R = np.zeros((2,2))
    R[0][0] =  np.cos(theta)
    R[0][1] = -np.sin(theta)
    R[1][0] =  np.sin(theta)
    R[1][1] =  np.cos(theta)
    return np.dot(R,[x,y])

def plotScatter(C, f_out="circle.pdf"):
    """
    Make scatter plot for a list of (x,y) value pairs
    """
    from matplotlib import pyplot as plt
    import matplotlib as mpl
    plt.clf()
    plt.style.use('ggplot')
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.scatter(C[:,0], C[:,1], marker=".")
    plt.savefig(f_out)



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



if __name__ == "__main__":
    import sys
    if len(sys.argv)!=4:
        print("Usage: {} NPOINTS PERCENTILE PLOTPREFIX".format(sys.argv[0]))
        sys.exit(1)
    NP = int(sys.argv[1])
    PERC = float(sys.argv[2])
    prefix = sys.argv[3]

    np.random.seed(1000)
    # Randomly sample points in a circle
    C = np.array([rdmCircle(np.random.rand()) for _ in range(NP)])
    plotScatter(C, "{}_Rcircle.pdf".format(prefix))

    # Rotate and stretch those points, i.e. make ellipsis --- origin is shifted to (1,-3)
    ORI = np.array([1, -3])
    E = np.array([ORI + elliptify(c[0], c[1], 9, 2, theta=np.pi/3) for c in C])


    # Eigen tune construction
    ET, CTR = construct(E, plot_prefix=prefix, percentile=PERC)

