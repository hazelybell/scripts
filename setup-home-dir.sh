#!/bin/bash

if [[ ! -e ~/scripts/bashrc ]]
then	echo "Expected scripts to be in ~/scripts"
	exit 1
fi

cd "$HOME"
function lns() {
	ln --backup=numbered --suffix=.old \
		--verbose --symbolic "$@"
}

function replace() {
	if [[ -z "$2" ]]
	then	source="scripts/$1"
		destination="$1"
	else	source="scripts/$1"
		destination="$2"
	fi
	if [[ ! -e $source ]]
	then echo "$source is missing!"
	fi
	if [[ -e "$destination" ]]
	then if [[ -h "$destination" ]]
		then rm -vf "$destination"
		else lns "$source" "$destination"
		fi
	fi
}

replace .inputrc
replace .toprc
replace .screenrc
replace .bashrc
replace konsolerc .config/konsolerc
replace konsole .local/share/konsole
