#! /usr/bin/env bash

for vel in  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 5 10
do
    ./fitting_group.py -v $vel
done
