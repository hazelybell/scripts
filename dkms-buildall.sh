#!/bin/bash

for i in `ls /var/lib/initramfs-tools`
do	for j in `find /var/lib/dkms -maxdepth 2 -mindepth 2 -type d -printf '%P '`
	do	dkms install $j -k $i
	done
done
