#! /usr/bin/env bash

# clean up after past runs
rm temp_plots/*.png

rm myvideo.avi
rm myvideo_slower.avi
rm myvideo2.avi
rm myvideo2_slower.avi

python traceback_plotter.py $1

ffmpeg -f image2 -i temp_plots/%dplotXY.png -vcodec mpeg4 -b 800k myvideo2XY.avi
ffmpeg -f image2 -i temp_plots/%dplotXZ.png -vcodec mpeg4 -b 800k myvideo2XZ.avi
ffmpeg -f image2 -i temp_plots/%dplotYZ.png -vcodec mpeg4 -b 800k myvideo2YZ.avi

# ffmpeg -i temp_plots/%dplotXY.png -vcodec mpeg4 myvideoXY.avi
#ffmpeg -i myvideo.avi -filter:v "setpts=2.0*PTS" myvideo_slower.avi

# ffmpeg -i myvideo.avi -filter:v "setpts=2.0*PTS" -vcodec mpeg4 -b 800k myvideo2_slower.avi
