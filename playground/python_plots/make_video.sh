#! /usr/bin/env bash
# rm plot_*.jpg
# python make_plots.py
ffmpeg -i plot_%d.jpg -vcodec mpeg4 test3.avi
