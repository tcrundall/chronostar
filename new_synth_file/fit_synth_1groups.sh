#!/usr/bin/env bash
./fit_groups.py -b 500 -p 1000 -i data/tb_synth_data_1groups_20stars.pkl -l
./fit_groups.py -b 500 -p 1000 -i data/tb_synth_data_1groups_30stars.pkl -l
./fit_groups.py -b 500 -p 1000 -i data/tb_synth_data_1groups_40stars.pkl -l
./fit_groups.py -b 500 -p 1000 -i data/tb_synth_data_1groups_50stars.pkl -l
./fit_groups.py -b 500 -p 1000 -i data/tb_synth_data_1groups_60stars.pkl -l
