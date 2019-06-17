STEMS='XY XZ XU YV ZW YZ'
# STEMS='XZ'
for STEM in $STEMS; do
  COUNTER_LENGTH=3
  PLOTREGEX=${COUNTER_LENGTH}d_$STEM.png
  ffmpeg -y -i %$PLOTREGEX -c:v libx264 -vf fps=15 -pix_fmt yuv420p -vf reverse $STEM.mp4
done

# ffmpeg -i frame-%d.jpg -vf reverse reversed.mp4
