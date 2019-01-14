#!/usr/bin/env python

import numpy as np
import scipy.stats as st

np.random.seed(1000)

def rdmCircle(rho=1):
    """
    Sample a point on a circle
    """
    phi = 2 * np.pi * np.random.rand()
    x = np.sqrt(rho) * np.cos(phi)
    y = np.sqrt(rho) * np.sin(phi)
    return x, y

def rdmEllipse(a,b, theta=0.5, rho=1):
    """
    Sample a point on a circle
    """
    phi = 2 * np.pi * np.random.rand()
    x = np.sqrt(rho) * np.cos(phi)
    y = np.sqrt(rho) * np.sin(phi)
    x*=a/2.
    y*=b/2.

    R = np.zeros((2,2))
    R[0][0] =  np.cos(theta)
    R[0][1] = -np.sin(theta)
    R[1][0] =  np.sin(theta)
    R[1][1] =  np.cos(theta)
    return np.dot(R,[x,y])

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

def mkEllipsis2D(a, b, phi):
    """
    """
    import scipy.stats as st
    _yvals =  st.multivariate_normal(yvals, C).rvs(size=1)


# Generate some points on the unit circle
np.random.seed(1000)
C = np.array([rdmCircle() for _ in range(1000)])
plotScatter(C, "circle.pdf")

# Rotate and stretch those points, i.e. make ellipsis
# E = np.array([elliptify(c[0], c[1], 3, 2, theta=0) for c in C])
E = np.array([elliptify(c[0], c[1], 9, 2, theta=np.pi/3) for c in C])

plotScatter(E, "ellipsis.pdf")

cov_E = np.cov(E.T)

S, V = np.linalg.eig(cov_E)
print(S)

# EV = np.array([np.diag(np.sqrt(S))@V.T@e for e in E])
EV = np.array([V.T@e for e in E])
# EV = np.array([e@V.T@np.diag(np.sqrt(S))@V for e in E])
plotScatter(EV, "ellipsisV.pdf")

# EVK = np.array([np.diag([np.sqrt(1/S[0]), np.sqrt(1/S[1])])@e for e in EV])
# EVK = np.array([np.diag([0.5/np.sqrt(S[0]), 0.5/np.sqrt(S[1])])@e for e in EV])
EVK = np.array([np.diag(0.5/np.sqrt(S))@V.T@e for e in E])
plotScatter(EVK, "ellipsisVK.pdf")




np.random.seed(1000)
C = np.array([rdmCircle(np.random.rand()) for _ in range(1000)])
plotScatter(C, "Rcircle.pdf")

# Rotate and stretch those points, i.e. make ellipsis
# E = np.array([elliptify(c[0], c[1], 3, 2, theta=0) for c in C])
E = np.array([elliptify(c[0], c[1], 9, 2, theta=np.pi/3) for c in C])

plotScatter(E, "Rellipsis.pdf")

cov_E = np.cov(E.T)

S, V = np.linalg.eig(cov_E)
print(S)

# EV = np.array([np.diag(np.sqrt(S))@V.T@e for e in E])
EV = np.array([V.T@e for e in E])
# EV = np.array([e@V.T@np.diag(np.sqrt(S))@V for e in E])
plotScatter(EV, "RellipsisV.pdf")

# EVK = np.array([np.diag([np.sqrt(1/S[0]), np.sqrt(1/S[1])])@e for e in EV])
# EVK = np.array([np.diag([0.5/np.sqrt(S[0]), 0.5/np.sqrt(S[1])])@e for e in EV])
# EVK = np.array([np.diag(0.5/np.sqrt(S))@V.T@e for e in E])
EVK = np.array([np.diag(0.5/np.sqrt(S))@V.T@e for e in E])
plotScatter(EVK, "RellipsisVK.pdf")


intercept = np.percentile(np.linalg.norm(EVK, axis=1), 68)

good = np.where ( np.linalg.norm(EVK, axis=1) < intercept )
bad  = np.where ( np.linalg.norm(EVK, axis=1) > intercept )
plotScatter(E[good], "Rgood.pdf")
plotScatter(E[bad],  "Rbad.pdf")


# Eigentune coordinates in the unit sphere world
dim=2
UET = intercept*np.append(-np.eye(dim), np.eye(dim), axis=1).reshape((2*dim,2))

# Note that we now need to apply the inverse transforms to get the real world coordinates

# Stretch
SET = np.array([np.diag(1/(0.5/np.sqrt(S)))@u for u in UET])

# Rotate
ET = np.array([V@s for s in SET])

plotScatter(ET, "eigentunes.pdf")

from matplotlib import pyplot as plt
import matplotlib as mpl
plt.clf()
plt.style.use('ggplot')
fig,ax = plt.subplots(1)
ax.set_aspect('equal')
ax.scatter(E[:,0],  E[:,1],  c="b", marker=".")
ax.scatter(ET[:,0], ET[:,1], c="r", marker="x")
plt.savefig("summary.pdf")

from IPython import embed
embed()
