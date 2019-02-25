#!/bin/bash
if [[ -z "$1" ]]
then	echo "Usage: $0 destination"
	exit 1
fi
DEST="$1"
bind () {
	# --rbind and --make-rslave have to be seperate for mount <2.33
	mount --rbind /"$1" "$DEST"/"$1" && mount --make-rprivate "$DEST"/"$1"
}

set -o xtrace

bind dev
bind proc
bind sys

# LVM fix -- LVM/grub/initramfs takes forever without this
mkdir -p "$DEST"/run/udev && (
	bind run/udev
)

# dpkg script fix
umount --recursive --lazy "$DEST"/sys/fs/cgroup

read -p "Bind /boot inside the chroot? " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then	bind boot
fi

chroot $DEST

umount --recursive --lazy $DEST/dev
umount --recursive --lazy $DEST/proc
umount --recursive --lazy $DEST/sys
umount --recursive --lazy $DEST/run/udev
umount --recursive --lazy $DEST/boot


