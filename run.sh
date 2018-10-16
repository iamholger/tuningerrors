#!/bin/bash

NBINS=$1
PERC=$2

mpirun -np 8 python ../toy-study.py -s 100 -n 10000 -b ${NBINS} -t ${PERC} -c none -o phi2-${NBINS}-${PERC}-none.json -p
mpirun -np 8 python ../toy-study.py -s 100 -n 10000 -b ${NBINS} -t ${PERC} -c sane -o phi2-${NBINS}-${PERC}-sane.json -p
mpirun -np 8 python ../toy-study.py -s 100 -n 10000 -b ${NBINS} -t ${PERC} -c mad  -o phi2-${NBINS}-${PERC}-mad.json  -p

pdfmerge -o MGNone.pdf *none*pdf
pdfmerge -o MGSane.pdf *sane*pdf
pdfmerge -o MGMad.pdf  *mad*pdf
