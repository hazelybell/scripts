#!/bin/bash

# init $PATH
source /etc/profile
RSYNC=/home/wz/scripts/rsync-wrapper.sh

echo "decrypting"
/sbin/cryptsetup luksOpen /dev/mapper/vg.cherry-rootbackup rootbackup -d /root/.rootbackup
if [[ -e /dev/mapper/rootbackup ]]
then	echo "decrypted"
else	exit 1
fi

mount -o noatime,compress=zlib /dev/mapper/rootbackup /mnt/rootbackup
if grep -q /mnt/rootbackup /proc/mounts
then 	echo "mounted"
else 	exit 2
fi

echo "taking snapshot"
btrfs subvolume snapshot -r / /rootbackup-snapshot
if [[ -d /rootbackup-snapshot/etc ]]
then	echo "took snapshot"
else 	exit 3
fi

$RSYNC -aP --delete-during //rootbackup-snapshot/ /mnt/rootbackup/

echo "exited slice... cleaning up"

btrfs subvolume delete /rootbackup-snapshot
umount /mnt/rootbackup
/sbin/cryptsetup luksClose rootbackup
