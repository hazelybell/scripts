#!/bin/bash

if [[ ! -e ~/scripts/bashrc ]]
then	echo "Expected scripts to be in ~/scripts"
	exit 1
fi

cd "$HOME"
function lns() {
	ln --backup=numbered --suffix=.old \
		--verbose --symbolic  --relative \
		--no-target-directory "$@"
}
function mvb() {
	mv --backup=numbered --suffix=.old \
		--verbose \
		--no-target-directory "$@"
}

function replace() {
	if [[ -z "$2" ]]
	then	source="scripts/$1"
		destination="$1"
	else	source="scripts/$1"
		destination="$2"
	fi
	if [[ ! -e $source ]]
	then	echo "$source is missing!"
		return
	fi
	if [[ -h "$destination" ]]
	then	if [[ "$destination" -ef "$source" ]]
		then	rm -f "$destination"
		fi
	else 	if [[ -d "$destination" ]]
		then 	# it is a directory and not a symbolic link
			mvb "$destination" "$destination".old
		fi
	fi
	lns "$source" "$destination"
}

replace .inputrc
replace .toprc
replace .screenrc
replace .bashrc
replace .gitconfig
replace konsolerc .config/konsolerc
replace konsole .local/share/konsole
