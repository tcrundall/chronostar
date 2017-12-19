#! /usr/bin/env bash

# clean up after past runs
# rm temp_plots/*.png

# rm myvideo.avi
# rm myvideo_slower.avi
# rm myvideo2.avi
# rm myvideo2_slower.avi

# python traceback_plotter.py $1

./generate_uv_gif.py
#rm uv_gif/uv.avi
#rm uv_gif/uv_slow.avi
#rm xy_gif/xy.avi
rm combined_gif/*.avi

#ffmpeg -f image2 -i uv_gif/%d.png -vcodec mpeg4 -b 800k uv_gif/uv.avi
#ffmpeg -f image2 -i xy_gif/%d.png -vcodec mpeg4 -b 800k xy_gif/xy.avi


ffmpeg -f image2 -i combined_gif/%d.png -vcodec mpeg4 -b 800k combined_gif/combined.avi
