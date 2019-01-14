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


### These bits wre just for testing
# # Generate some points on the unit circle
# np.random.seed(1000)
# C = np.array([rdmCircle() for _ in range(1000)])
# plotScatter(C, "circle.pdf")

# # Rotate and stretch those points, i.e. make ellipsis
# E = np.array([elliptify(c[0], c[1], 9, 2, theta=np.pi/3) for c in C])

# plotScatter(E, "ellipsis.pdf")

# cov_E = np.cov(E.T)

# S, V = np.linalg.eig(cov_E)
# print(S)

# EV = np.array([V.T@e for e in E])
# plotScatter(EV, "ellipsisV.pdf")

# EVK = np.array([np.diag(0.5/np.sqrt(S))@V.T@e for e in E])
# plotScatter(EVK, "ellipsisVK.pdf")


# Randomly sample points in a circle
np.random.seed(1000)
C = np.array([rdmCircle(np.random.rand()) for _ in range(1000)])
plotScatter(C, "Rcircle.pdf")

# Rotate and stretch those points, i.e. make ellipsis --- origin is shifted to (1,-3)
ORI = np.array([1, -3])
E = np.array([ORI + elliptify(c[0], c[1], 9, 2, theta=np.pi/3) for c in C])
plotScatter(E, "Rellipsis.pdf")

# The center of the ellipsoid --- needed for translation operation
CTR = np.mean(E, axis=0)
plotScatter(E-CTR, "RellipsisOri.pdf")

# The covariance matrix --- needed to do rotation
cov_E = np.cov(E.T)

# Eigenvalue problem to get rotation matrix, V
S, V = np.linalg.eig(cov_E)

# Apply rotation and translation
EV = np.array([V.T@e for e in E-CTR])
plotScatter(EV, "RellipsisV.pdf")

# Apply stretch, rotation and translation so that all points (EVK) are in unit sphere
EVK = np.array([np.diag(0.5/np.sqrt(S))@V.T@e for e in E-CTR])
plotScatter(EVK, "RellipsisVK.pdf")

# Calculate the radius corresponding to the percentile in question
intercept = np.percentile(np.linalg.norm(EVK, axis=1), 68)

# Just some control plots
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

# Rotate and translate
ET = np.array([V@s + CTR for s in SET])

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
