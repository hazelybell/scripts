#!/bin/bash
xrandr \
	--output eDP-1-1 --primary --mode 1920x1080 --dpi 180 \
	--output HDMI-0 --mode 1920x1080 --dpi 180 --same-as eDP-1-1 \
	--output DP-1 --mode 1920x1080 --dpi 180 --above eDP-1-1
