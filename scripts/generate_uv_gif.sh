#! /usr/bin/env bash

./generate_uv_gif.py
rm combined_gif/*.avi

ffmpeg -f image2 -i combined_gif/%d.png -vcodec mpeg4 -b 800k combined_gif/combined.avi
