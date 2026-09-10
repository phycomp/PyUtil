#!/usr/bin/bash
alias pacins='sudo pacman -Syu --noconfirm'
alias pkginfo='pacman -Si'
alias Search='pacman -Ss'
alias Update='sudo pacman -Syy'
alias Remove='sudo pacman -R --noconfirm'
#alias Remove='sudo pacman --remove --noconfirm'
alias rfrshKey='sudo pacman-key --refresh-keys'
alias pacListQuery='pacman -Ql'
alias ownQuery='pacman -Qo'
alias pacRegex='pacman -Qs'
alias LstPckg='pacman -Qe'
alias pinst='pip install --upgrade'
alias pyinst='pypy3 -m pip install --upgrade pip'
alias lstrm='ls -trm'
alias rm='rm -rf'
alias cd-='cd -'
alias cd..='cd ..'
alias act='source ~/flex/bin/activate'
alias luzp='unzip -l -O big5'
alias b5uzp='unzip -O big5'
alias ztvf='tar -ztvf'
alias zxvf='tar -zxvf'
#alias zjvf='tar -zjvf'
alias Jtvf='tar -Jtvf'
alias Jxvf='tar -Jxvf'
alias jtvf='tar -jtvf'
alias jxvf='tar -jxvf'
alias getNvim='wget https://github.com/neovim/neovim/releases/download/nightly/nvim-linux64.tar.gz'
alias noget='wget --no-check-certificate'
alias pdf2raw='pdftotext -layout -raw -nopgbrk'
alias tmxttch='tmux attach -t'
alias tsssn='tmux list-sessions'
alias tmxnws='tmux new -s'
alias stdtime='sudo ntpdate tock.stdtime.gov.tw'       #ntptock
alias rgpy="rg -tpy -g '!pythnINS' -g'!Python' -g '!fhirconverter' -g '!transformer' -g '!Downloads'"
alias rgmd="rg -tmd -g '!pythnINS' -g'!Python' -g '!fhirconverter' -g '!transformer' -g '!Downloads'"
nvm() {
  nvim -S $HOME/.config/nvim/sssn/$1    #.sssn
}

dfile() {
  du -a /home/josh/$1/* | sort -nr | head -n100
}

rgSrch() {
  if [[ $# -eq 3 ]]; then
    rg -t$1 -g '!pythnINS' -g'!Python' -g '!fhirconverter' -g '!transformer' -g '!Downloads' $2 $3
  elif [[ $# -eq 2 ]]; then
    rg -tpy -g '!pythnINS' -g'!Python' -g '!fhirconverter' -g '!transformer' -g '!Downloads' $1 $2
  elif [[ $# -eq 1 ]]; then
    rg -tpy -g '!pythnINS' -g'!Python' -g '!fhirconverter' -g '!transformer' -g '!Downloads' $2
  fi
}

ntScrll() {
  mouseID=`xinput list|awk '/Mouse.*id=/ {print $8}'|awk '{gsub("id=", ""); print $0}'`
  ntscrllProp=`xinput list-props $mouseID |awk -e '/libinput Natural Scrolling Enabled \(.*\):/ {gsub("[():]", "");print $5}'`
  xinput set-prop $mouseID $ntscrllProp 1
}

lprRange() {
  lpr -o A4 -o fit-to-page -o media=A4 -o page-ranges=$1 $2
}

prntscrn() {
  tmpFile=/tmp/"$2".png
  import -window root -pause $1 $tmpFile
  gimp $tmpFile&
}
