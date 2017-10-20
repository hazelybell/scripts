#!/bin/bash
( pacmd list-sinks | grep -q jack ) || (
	pacmd unload-module module-jackdbus-detect ;
	sleep 1 ;
	pacmd load-module module-jackdbus-detect channels=2 ;
)
