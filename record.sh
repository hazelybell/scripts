#!/bin/bash
rm output.mkv
ffmpeg -f alsa -ac 1 -i default \
	-f x11grab -r 30 -s 1920x1080 -i :0.0 \
	-acodec libvorbis -vcodec libx264 \
	-preset veryfast -crf 10 -threads 0 -y output.mkv
