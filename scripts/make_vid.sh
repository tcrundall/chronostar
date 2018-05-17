ffmpeg -i temp_plots/%2d_xu_tf.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p traceforward_xu.mp4
ffmpeg -i temp_plots/%2d_yv_tf.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p traceforward_yv.mp4
ffmpeg -i temp_plots/%2d_zw_tf.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p traceforward_zw.mp4
ffmpeg -i temp_plots/%2d_xu_tb.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p traceback_xu.mp4
ffmpeg -i temp_plots/%2d_yv_tb.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p traceback_yv.mp4
ffmpeg -i temp_plots/%2d_zw_tb.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p traceback_zw.mp4
