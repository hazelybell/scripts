#!/bin/bash
xrandr \
	--output eDP-1-1 --primary --mode 1280x720 --dpi 120 \
	--output HDMI-1 --mode 1280x720 --dpi 120 --same-as eDP-1-1 \
	--output DP-1 --mode 1280x720 --dpi 120 --above eDP-1-1
