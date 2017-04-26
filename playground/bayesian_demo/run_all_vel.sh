#! /usr/bin/env bash

for vel in  0.1 0.2 0.5 1 2 5 10
do
    ./fitting_group.py -v $vel
done
