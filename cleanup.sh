#!/bin/bash
now=`date +'%s'`
nowtd="$HOME/`date +'%Y/%m/%d'`"
logto=.cleanuplog
DIRS="$HOME $HOME/Desktop $HOME/dwhelper"
shopt -s extglob
IGNORE='*@(bin|lib|man|share|include|borg|backup|mprime5|.DCOP*|.dbus|.config|.gtkrc-1.2-gnome2|.local|.openoffice.org*|.recently-used|.Trash|.xsession-errors*|.desktop|.directory|Music|unison.log|.git|.procmail.log|.Xauthority|.ICEauthority|.spamassassin|.gnome2_private|.gconfd|.fonts.conf|.gksu.lock|.gtk_qt_engine_rc|.gtkrc*|.mcop*|.nautilus|.texmf-var|porn|etch|gentoo|ubuntu|baal|muumuu)'
DELETE='*@(.fontconfig|.dvdcss|.evolution|.adobe|.macromedia|.thumbnails|.gnome|.icons|.metacity|.themes|.update-notifier)'
commit=$1

function maybe() {
    echo "$@" >>$logto
    if [ "$commit" != softly ] ; then
        echo "$@"
    fi
    if [ "$commit" = commit ] || [ "$commit" = softly ] ; then
         "$@"
    fi
}

function debug() {
#   echo "$@" >>$logto
    if [ "$commit" = debug ] ; then
        echo "$@"
    fi
}

for d in $DIRS ; do
    if [ -d $d ]; then
        cd $d
    else
        continue
    fi
    for f in * .* ; do
        if git-ls-files --error-unmatch $f &>/dev/null ; then
            debug git $f
            continue 1
        fi
        if [ -d "$f" ] ; then
            for di in $DIRS ; do
                case "$d/$f" in
                    "$d/.."|"$d/."|"$di")
                        debug subdir $f
                        continue 2
                    ;;
                esac
            done
        fi
        if [ -d "$f/.svn" ] ; then
            debug svn $f
            continue 1
        fi
        case $d/$f in
            $IGNORE)
                debug ignore $f
                continue 1
            ;;
            $DELETE)
                debug delete $f
                maybe rm -rvf $f
                continue 1
            ;;
            */[0-9][0-9][0-9][0-9])
                debug dated $f
                continue 1
            ;;
        esac
        modified=`stat -c %Z "$f"`
        if (($now - $modified > 30 * 86400)); then
            td="$HOME/`date \"--date=\`stat \"$f\" -c %z\`\" +'%Y/%m/%d'`"
            dest="$td/`basename \"$f\"`"
            debug "$dest"
            [ -d $td ] || maybe mkdir -p $td
            maybe mv -v "$f" "$dest"
        elif (($now - $modified > 7 * 86400)); then
            debug aging $f
            echo "${f} aging" $(( 30 - ($now - $modified) / 86400 )) days left
        else
            debug toonew $f
        fi
    done
done

if [ -f $logto ] ; then
    [ -d $nowtd ] || mkdir -p $nowtd
    mv $logto $nowtd/$logto
fi
