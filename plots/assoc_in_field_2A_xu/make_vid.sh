## ffmpeg -pattern_type glob -i '*xu.png' -c:v libx264 -vf fps=25 -pix_fmt yuv420p assoc_in_field_xu.mp4
## ffmpeg -i temp_plots/%2d_xu_tf.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p traceforward_xu.mp4
## ffmpeg -i 'iter_%2d_xu.pdf' -c:v libx264 -vf fps=25 -pix_fmt yuv420p assoc_in_field_xu.mp4

for old in *.pdf; do pdftoppm -png iter_84_xu.pdf > iter_84_xu.png; done
ffmpeg -i 'iter_%2d_xu.png' -vf fps=25  assoc_in_field_xu.mp4
for old in *.pdf; do pdftoppm -png $old > `basename $old .pdf`.png; done
