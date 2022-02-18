#! /usr/bin/env python

import numpy as np
import apprentice

def faster_chi(lW2, lY, lP, lE2, nb):
    s=0
    NOM = lW2*(lY - lP)**2
    DEN = lE2 + (0.05*lY)**2
    return np.sum(NOM/DEN)


def mkTrainingData(tobj, nsamples, fout, base_seed, nstart, nrestart):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    import sys
    if rank ==0:
        print("Training data generation")
        print("Will produce {} samples".format(nsamples*size))
        if size>1:
            print("Distributed amongst {} ranks".format(size))
        sys.stdout.flush()


    RESULTS = []
    if rank==0:
        RESULTS.append(tobj.minimize(nstart,nrestart))

    comm.barrier()

    import copy
    DATA = copy.deepcopy(tobj._Y)
    ERR2 = copy.deepcopy(tobj._E2)

    datacov = np.eye(len(ERR2)) * 1./ERR2

    import scipy.stats as st, scipy.optimize as opt
    np.random.seed(base_seed+rank)
    mn = st.multivariate_normal(DATA, datacov, allow_singular=True)

    samples = mn.rvs(size=nsamples)

    # from IPython import embed
    # embed()

    import time, datetime
    t0=time.time()
    tstart = datetime.datetime.fromtimestamp(t0)
    if rank==0:
        print("Start at {}".format( tstart.strftime('%Y-%m-%d %H:%M:%S') ))
    sys.stdout.flush()

    for n, smpl in enumerate(samples):
        tobj._Y = smpl
        RESULTS.append(tobj.minimize())
        now = time.time()
        tel = now - t0
        ttg = tel*(nsamples-n)/(n+1)
        eta = now + ttg
        eta = datetime.datetime.fromtimestamp(now + ttg)
        if rank==0:
            sys.stdout.write("[{}] {}/{} (elapsed: {:.1f}s, to go: {:.1f}s, ETA: {})\r".format(rank, n+1, nsamples, tel, ttg, eta.strftime('%Y-%m-%d %H:%M:%S')))
        sys.stdout.flush()

    allResults = comm.gather(RESULTS, root=0)



    if rank==0:
        t1=time.time()
        print("\n\nTotal run time: {} seconds".format(t1-t0))



        OUT = []
        for rr in allResults:
            for r in rr:
                temp = list(r.x)
                temp.append(r.fun/2.)
                OUT.append(temp)

        import json

        dd = {"DATA" : OUT,
                "NBINS": len(DATA),
                # "PARAMS": tobj._pnames
                }

        with open(fout, "w") as f:
            json.dump(dd, f, indent=4)

        print("Output written to {}".format(fout))

if __name__ == "__main__":
    import optparse, os, sys

    op = optparse.OptionParser(usage=__doc__)
    op.add_option("-v", "--debug", dest="DEBUG", action="store_true", default=False,
                  help="Turn on some debug messages")
    op.add_option("-q", "--quiet", dest="QUIET", action="store_true", default=False,
                  help="Turn off messages")
    op.add_option("-o", "--output", dest="OUTPUT",default="rbftraining.txt",
                  help="Training data output file (Default: %default)")
    op.add_option("-s", "--seed", dest="SEED", type=int, default=1234,
                  help="The random seed (Default: %default)")
    op.add_option("-d", "--data", dest="DATA", default=None,
                  help="The data file (Default: %default)")
    op.add_option("-D", "--design", dest="DESIGNSIZE", type=int, default=-1,
                  help="The size of the initial design (Default: %default)")
    op.add_option("-w", "--wfile", dest="WFILE", default=None,
                  help="The weight file for the inner optimisation (Default: %default)")
    op.add_option("--scan-n", dest="SCANNP", default=1, type=int,
                  help="Number of test points find a good migrad start point (default: %default)")
    op.add_option("--restart-n", dest="IORESTART", default=1, type=int,
                  help="Number of restart of the IO (default: %default)")
    opts, args = op.parse_args()

    OO = apprentice.appset.TuningObjective2(opts.WFILE, opts.DATA, args[0], debug=opts.DEBUG)
    mkTrainingData(OO, opts.DESIGNSIZE, opts.OUTPUT, opts.SEED, opts.SCANNP, opts.IORESTART)
