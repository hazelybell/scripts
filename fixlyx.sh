#!/bin/bash
export TFMFONTS='.;/home/warpzero/class/MATH584;{$TEXMF/fonts,$VARTEXFONTS}/tfm//'
export PATH="${PATH}:/home/warpzero/class/MATH584"
lyx "$@"
