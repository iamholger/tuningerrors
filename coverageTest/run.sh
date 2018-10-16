#!/bin/bash

NBINS=20
PERC=68.8


for n in {0..0}
do
    dname="run_${n}"
    echo $dname
    mkdir -p $dname
    cd $dname
        mpirun -np 7 python ../../toy-study.py -s 100 -n 10000 -b ${NBINS} -t ${PERC} -c sane -o phi2-${NBINS}-${PERC}-sane.json  --seed ${n} -q
    cd -
done
