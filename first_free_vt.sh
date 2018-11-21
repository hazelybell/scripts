#!/bin/bash
# Adapted from https://superuser.com/a/633185 2018-09-21
PIPE=$(mktemp -u)
mkfifo $PIPE
{
	sudo -E -- openvt --wait -- bash -c "tty >$PIPE"
} &
FREETTY=$(head -n1 $PIPE)
echo "Free TTY: $FREETTY"
rm -f $PIPE
unset PIPE
FREEVT=${FREETTY#/dev/tty}
echo "Free vt: $FREEVT"
