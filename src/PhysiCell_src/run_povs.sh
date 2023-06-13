#!/bin/bash

count=$(ls -1 ./output/*.svg 2>/dev/null | wc -l);

echo "number of files is $count" 
./povwriter 0:1:20;
for scene in *.pov; do povray +w640 +h640 +a0.3 -D $scene; done ;
ffmpeg -r 8 -f image2 -i ./pov%08d.png -vcodec libx264 -pix_fmt yuv420p -strict -2 -tune animation -crf 15 -acodec none ./out.mp4


