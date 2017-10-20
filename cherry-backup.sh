#!/bin/bash

# init $PATH
source /etc/profile

if grep -q $BASHPID -r /sys/fs/cgroup/systemd/backup.slice
then	echo "in slice"
	rsync -aP --delete-during /rootbackup-snapshot/ /mnt/rootbackup/
	echo "exiting slice"
else	echo "not in slice"

	echo "decrypting"
	cryptsetup luksOpen /dev/mapper/vg.cherry-rootbackup rootbackup -d /root/.rootbackup
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

	systemd-run --slice backup.slice --scope "$0" "$@"

	echo "exited slice... cleaning up"

	btrfs subvolume delete /rootbackup-snapshot
	umount /mnt/rootbackup
	cryptsetup luksClose rootbackup
fi


