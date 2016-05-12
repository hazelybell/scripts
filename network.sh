#!/bin/bash
if [ x"$1" = x"off" ]
then	echo Turning off network stuff
	killall -SIGSTOP konqueror 
	killall -SIGSTOP kontact
	killall -SIGSTOP npviewer.bin
	killall -SIGSTOP firefox-bin
	/etc/init.d/networking stop
	init 3
	modprobe -r tg3
	modprobe -r b43
elif [ x"$1" = x"on" ] 
then	echo Turning on network stuff
	modprobe tg3
	modprobe b43
	/etc/init.d/networking start
	init 2
	killall -SIGCONT konqueror 
	killall -SIGCONT kontact
	killall -SIGCONT firefox-bin
	killall -SIGCONT npviewer.bin
else
	echo "Usage: $0 off|on"
fi
