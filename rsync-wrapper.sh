#!/bin/bash

# init $PATH
source /etc/profile

if grep -q $BASHPID -r /sys/fs/cgroup/systemd/backup.slice
then    rsync "$@"
else    systemd-run --slice backup.slice --scope "$0" "$@"
fi
