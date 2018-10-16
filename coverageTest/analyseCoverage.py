#!/usr/bin/env python


def isCovered(fname, target=68.8):
    import json
    with open(fname) as f:
        jd=json.load(f)

    tunechi2 = jd["TUNECHI2"]
    vphi2    = jd["VPHI2"]
    return tunechi2 < np.percentile(vphi2, target)

import numpy as np

import os, sys

FIN=sys.argv[2:]
target=float(sys.argv[1])

CT = [int(isCovered(f, target)) for f in FIN]

print("Coverage: %.1f %%"%(100.*sum(CT)/len(CT)))
