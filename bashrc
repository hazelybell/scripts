# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

export PATH="$PATH:$HOME/scripts"
export PERL5LIB=$HOME/.perl:$HOME/.perl/lib/perl5:$PERL5LIB

# COMMANDS ONLY FOR THE INTERACTIVE SHELL FOLLOW -----------------------------

# Check if shell is interactive.
if [[ $- != *i* ]]; then
    return
fi

shopt -s histverify extglob cdspell cmdhist histappend dirspell
set -o noclobber
#set -o nounset
set -o pipefail

HISTFILESIZE=
HISTSIZE=
HISTCONTROL=ignoreboth

# User specific aliases and functions

export EDITOR="nano"
export PAGER="less"

alias cp="cp -i"
alias rm="rm -i"
alias mv="mv -i"
alias nano="nano -w"
#alias grep="grep --color=auto"
#alias ls="ls --color=auto"
#alias ll="ls -lhA --color=auto"
alias ll="ls -lhA"
alias gpg="gpg2"

# Keep your Home Directory Organized from http://www.splitbrain.org/blog/2008-02/27-keeping_your_home_directory_organized
export TD="$HOME/`date +'%Y/%m/%d'`"
td(){
    if [[ -z "$1" ]]; then
        td=$TD
        mkdir -p $td || return $?
    else
        td="$HOME/`date -d "$1 days" +'%Y/%m/%d'`";
    fi
    cd $td || return $?
    unset td
}

# Adapted from http://phraktured.net/config/.bashrc
function mkcd() { mkdir -p "$1" && cd "$1"; }
#function calc(){ awk "BEGIN{ print $* }" ;}
function calc() { 
  if [[ -x $(which gp) ]]; then
    echo "$@" | gp -q
    return
  fi
  if [[ -x $(which perl) ]]; then
    perl -le" print $* ;"
    return
  fi
  if [[ -x $(which awk) ]]; then
    awk "BEGIN{ print $* }"
    return
  fi
  eval "echo \$(($*))"
}
function pe { perl -le" print $* ;"; }
function hex2dec { awk 'BEGIN { printf "%d\n",0x$1}'; }
function dec2hex { awk 'BEGIN { printf "%x\n",$1}'; }
function mktar() { echo tar cJvf "${1%%/}.tar.xz" "${1%%/}/"; }
function own() { sudo chown -Rc ${USER} ${1:-.}; }
function rot13 () { echo "$@" | tr a-zA-Z n-za-mN-ZA-M; }
function gril() { grep -rl "$@" .; }
function grepword() { grep -Hnr "$@" .; }
function sendkey () {
    if [ $# -eq 1 ]; then
        local key=""
        if [ -f ~/.ssh/id_dsa.pub ]; then
            key=~/.ssh/id_dsa.pub
        elif [ -f ~/.ssh/id_rsa.pub ]; then
            key=~/.ssh/id_rsa.pub
        else
            echo "No public key found" >&2
            return 1
        fi
        ssh $1 'cat >> ~/.ssh/authorized_keys' < $key
    fi
}

function copy () { rsync -aP "$@"; }

function cargol {
    cargo --color always "$@" |& less -r
}

# COMMANDS ONLY FOR TELETYPES ------------------------------------------------

# Check if stdout/stderr is a teletype
if [[ ! ( -t 0 && -t 1 ) ]]; then
  return
fi

# Functions only for interactive sessions
function scp() { echo stop using scp; echo rsync -aP "$@"; }
function python() { echo "Which python?" >&2; }

# Check if we can start screen
if [[ "$TERM" != screen* ]] && [[ -v SSH_CONNECTION ]]; then
        exec screen -S remote -xRR
fi

shopt -s checkwinsize

# Get \l
TITLE_DEV="$(readlink /proc/self/fd/0)"
TITLE_DEV=${TITLE_DEV##*/}

# Sets the window title - adapted from redhat
case $TERM in
    xterm*)
        TITLE_FANCY='\033]0;${TITLE_COMMAND} (${USER}@${HOSTNAME%%.*}#${TITLE_DEV} ${PWD/#$HOME/~})\007'
        TITLE_TRAP=TITLE_MAKE
        PS1_TITLE='\[\033]0; \u@\h#\l \W\007\]'
        if [[ ! -v COLORTERM ]]; then # assume this is really a color terminal
          COLORTERM="$TERM"
        fi
        ;;
    screen*)
        TITLE_FANCY='\033k${TITLE_COMMAND} (${PWD/#$HOME/~})\033\\'
        TITLE_TRAP=TITLE_MAKE
        PS1_TITLE='\[\033k\W\033\\\]\[\033_\u@\h#\l\033\\\]'
        if [[ ! -v COLORTERM ]]; then # assume this is really a color terminal
          COLORTERM="$TERM"
        fi
        ;;
    linux)
        # Don't set TITLE_TRAP
        PS1_TITLE=''
        if [[ ! -v COLORTERM ]]; then # really a color terminal but doesn't set COLORTERM
          COLORTERM="$TERM"
        fi
        ;;
    *)
        # Don't set TITLE_TRAP
        PS1_TITLE=''
        ;;
esac

# Determine if we have a color terminal not found above
# Normally we could just do this but no one's ssh server is setup to AcceptEnv COLORTERM
if [[ ! -v COLORTERM ]]; then
    PROMPT_COMMAND=PS1_EXTRA
    export LESS="aMQSj.5"
else
    PROMPT_COMMAND=PS1_MAKE_COLOR

    # Make ls colorful
    if [[ -x $(which dircolors) ]]; then
        eval "$(dircolors -b)"
    fi
    alias grep="grep --color=auto"
    alias ls="ls --color=auto"
    alias ll="ls -lhA --color=auto"
    export LESS="aMQSRj.5" # Allow less to put out colors
    #alias lessc /usr/share/vim/vim72/macros/less.sh
fi

if [[ ${EUID} == 0 ]] ; then
    # Root prompt
    PS1=${PS1_TITLE}'$? \h \W \!\$ '
    PS1_F=${PS1_TITLE}'\[\033[01;33m\]$? \[\033[01;31m\]\h \[\033[01;33m\]\W \!\$\[\033[00m\] '
    PS1_T=${PS1_TITLE}'\[\033[01;32m\]$? \[\033[01;31m\]\h \[\033[01;33m\]\W \!\$\[\033[00m\] '
else
    # User prompt
    PS1=${PS1_TITLE}'$? \u@\h \W \!\$ '
    PS1_F=${PS1_TITLE}'\[\033[01;33m\]$? \[\033[01;35m\]\u@\h \[\033[01;32m\]\W \!\$\[\033[00m\] '
    PS1_T=${PS1_TITLE}'\[\033[01;32m\]$? \[\033[01;35m\]\u@\h \[\033[01;32m\]\W \!\$\[\033[00m\] '
fi

PS1_EXTRA() {
    history -a
}

PS1_MAKE_COLOR() {
    if [[ $? -ne 0 ]]; then
        PS1=${PS1_F}
    else
        PS1=${PS1_T}
    fi
    PS1_EXTRA
}

TITLE_MAKE() {
    TITLE_COMMAND=$BASH_COMMAND
    case ${TITLE_COMMAND} in
        PS1_MAKE_COLOR)
        ;;
        'TITLE_COMMAND=$BASH_COMMAND')
            trap - DEBUG
        ;;
        *)
            TITLE_COMMAND=${TITLE_COMMAND//--color=auto /} # Extraneous junk from aliases
            TITLE_COMMAND=${TITLE_COMMAND//[^[:print:]]/" "} # Convert all non-printing characters to spaces incl. newlines/tabs
            TITLE_COMMAND=${TITLE_COMMAND:0:30}
            TITLE_COMMAND=${TITLE_COMMAND//\\/\\\\} # Prevent stupid escapes...
#             echo -ne '\033[00m'
            eval "echo -ne \"$TITLE_FANCY\""
        ;;
    esac
    unset TITLE_COMMAND
}

# Dynamic title setting adapted from http://www.davidpashley.com/articles/xterm-titles-with-bash.html
if [[ -n ${TITLE_TRAP+x} ]] ; then
  trap "$TITLE_TRAP" DEBUG
fi

if [[ -x $(which lesspipe) ]]; then
  eval "$(lesspipe)"
fi

if [[ -e ~/scripts/bashrc-local ]]; then
  source ~/scripts/bashrc-local
fi
