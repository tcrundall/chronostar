#! /usr/bin/env bash
# Simple script to run a fixed age fit for a range of ages
for AGE in {0..30}
do
    echo "Fitting for $AGE Myr"
    ./fitting_BPMG.py -b 2000 -p 2000 -l -a $AGE
done
