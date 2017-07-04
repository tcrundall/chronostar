#! /usr/bin/env bash
for AGE in {0..30}
do
    echo "Fitting for $AGE Myr"
    ./fitting_BPMG.py -b 2000 -p 2000 -l -a $AGE
done
