alias getNvim='wget https://github.com/neovim/neovim/releases/download/nightly/nvim-linux64.tar.gz'
alias noget='wget --no-check-certificate'
alias bnzp='unzip -O big5'
alias rgpy="rg -tpy -g '!pythnINS' -g'!Python' -g '!fhirconverter' -g '!transformer' -g '!Downloads'"
alias rgmd="rg -tmd -g '!pythnINS' -g'!Python' -g '!fhirconverter' -g '!transformer' -g '!Downloads'"
alias Install='sudo pacman -Syu --noconfirm'
alias sffc='libreoffice --impress'
alias pacListQuery='pacman -Ql'
alias ownQuery='pacman -Qo'
alias nbcnvrt='jupyter-nbconvert --to python'
alias pacRegex='pacman -Qs'
alias pkginfo='pacman -Si'
alias autosub='autosub -S zh-TW -D zh-TW'
#alias Remove='sudo pacman -R --noconfirm'
alias Remove='sudo pacman --noconfirm --remove'
alias Search='pacman -Ss'
alias LstPckg='pacman -Qe'
#alias ntscrll='xinput set-prop 11 281 1'
#alias ntscrll='xinput set-prop 10 307 1'
alias lpr='lpr -o A4 -o fit-to-page -o media=A4'
alias cd..='cd ..'
alias strmltRun='streamlit run'
alias psgrep='ps aux|grep'
alias cd-='cd -'
alias lsrtm='ls -rtm'
alias ls='ls --color=auto'
alias ztvf='tar -ztvf'
alias zxvf='tar -zxvf'
alias Jtvf='tar -Jtvf'
#alias Jxvf='tar Jxvf'
alias Jxvf='tar -Jxvf'
alias jtvf='tar -jtvf'
alias jxvf='tar -jxvf'
alias rm='rm -rf'
alias chx='chmod +x'
alias lnzp='unzip -l -O big5'
alias cnvt2txt='soffice --headless --convert-to txt'
alias runserver='./manage.py runserver'
alias startapp='./manage.py startapp'
alias index_update='./manage.py update_index'
alias rebuild_index='./manage.py rebuild_index'
alias migrate='./manage.py migrate'
alias mkmigration='./manage.py makemigrations'
alias createsuperuser='./manage.py createsuperuser'
alias djshell='./manage.py shell'
alias dbshell='./manage.py dbshell'
alias dbeaver="env LANG=C $HOME/dbeaver/dbeaver -vm $HOME/jre1.8.0_202/bin/"
alias mplyr='mplayer -fs -shuffle -playlist'
alias pdf2raw='pdftotext -layout -raw -nopgbrk'
alias ctlrestart='sudo systemctl restart'
alias ctlenable='sudo systemctl enable'
alias ctlstatus='systemctl status'
alias fuserMnt='sudo chmod u+s `which fusermount`'
#alias less='less -R'
alias grep='/usr/bin/grep --color=always'
#alias grep="grep --color=always"
#export GREP_OPTIONS="--color=always"
#alias nclrgrep='/usr/bin/grep --color=none'
alias pushGit='git push -u origin master'
alias cloneGit='git clone'
#alias batti='acpi -b'
alias mplayer='mplayer -fs -vo x11'
#alias activate='source bin/activate'
alias mv='mv -i'
alias gitpush='git push origin master'
#alias inplace='python setup.py build_ext --inplace'
alias pipinstall='pip install --upgrade'
alias pipunstall='pip uninstall'
#alias sshCancer='ssh -p 4822 cancer'
#alias bgcftp='ssh bigdata@bgcftp'
#alias sshredcap='ssh -p 5722 redcappg@60.251.7.157'
#alias db2ssh='ssh vghiscap@db2'
#alias ssh='ssh -X'
alias ssh='ssh -X -Y'
alias sshNode='env LD_LIBRARY_PATH=: ssh -X -Y'
alias scpNode='env LD_LIBRARY_PATH=: scp'
alias nodeSSH='env LD_LIBRARY_PATH=: ssh -X -Y'
alias nodeSCP='env LD_LIBRARY_PATH=: scp'
#alias R='env LANG=C R'
#alias aktivate="source $HOME/bdata2/bin/activate"
#alias Ackivate="source bin/activate"
#alias ackivate2="source $HOME/bdata2/bin/activate"
alias stdtime='sudo ntpdate tock.stdtime.gov.tw'       #ntptock
#alias stdtime='sudo ntpdate time.stdtime.gov.tw'
#tock.stdtime.gov.tw #watch.stdtime.gov.tw #time.stdtime.gov.tw #clock.stdtime.gov.tw	#tick.stdtime.gov.tw
alias utf8='iconv -fbig5 -tutf8'
#alias refreshKeys='sudo pacman-key --refresh-keys'
#alias pipsrch='pip search'
alias rpmExtract="rpm2cpio $1 | cpio -idmv"
alias runflask='flask run'
alias fabflask='flask fab'
alias b2u='convmv -f big5 -t utf8 -r --notest *'
alias rtrvffce='soffice --headless --convert-to txt:Text'
alias setLD='env LD_LIBRARY_PATH=.'
#alias ytbdlSbttl='youtube-dl --sub-lang zh-Hant --write-auto-sub --skip-download'
#alias ytbdlMP3='youtube-dl --extract-audio --audio-format mp3'
alias tmxttch='tmux attach -t'
alias tsssn='tmux list-sessions'
alias tmxnws='tmux new -s'
alias curl='curl -LO'
alias spyder=/home/josh/.local/spyder-6/envs/spyder-runtime/bin/spyder
alias uninstall-spyder=/home/josh/.local/spyder-6/uninstall-spyder.sh

condaenv(){
  __conda_setup="$('/home/josh/.local/spyder-6/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
      eval "$__conda_setup"
  else
      if [ -f "/home/josh/.local/spyder-6/etc/profile.d/conda.sh" ]; then
          . "/home/josh/.local/spyder-6/etc/profile.d/conda.sh"
      else
          export PATH="/home/josh/.local/spyder-6/bin:$PATH"
      fi
  fi
  unset __conda_setup
}

dfile(){
  du -a /home/josh/$1/* | sort -nr | head -n100
}

nvm()
{
  nvim -S $HOME/.config/nvim/sssn/$1	#.sssn
}

setSSLcert()
{
#prmpt=`basename $VIRTUAL_ENV_PROMPT`
#extension="${prmpt##*.}"
#export SSL_CERT_FILE="$VIRTUAL_ENV/lib/python3.$extension/site-packages/certifi/cacert.pem"
if [ -z $VIRTUAL_ENV_PROMPT ]; then
        export SSL_CERT_FILE=$HOME/pythnINS/lib/python$trlPyVer/site-packages/certifi/cacert.pem
else
        export trlPyVer=${VIRTUAL_ENV_PROMPT: -4}
        export SSL_CERT_FILE=$VIRTUAL_ENV/lib/python$trlPyVer/site-packages/certifi/cacert.pem
fi
}

rgSrch()
{
  if [[ $# -eq 3 ]]; then
    rg -t$1 -g '!pythnINS' -g'!Python' -g '!fhirconverter' -g '!transformer' -g '!Downloads' $2 $3
  elif [[ $# -eq 2 ]]; then
    rg -tpy -g '!pythnINS' -g'!Python' -g '!fhirconverter' -g '!transformer' -g '!Downloads' $1 $2
  elif [[ $# -eq 1 ]]; then
    rg -tpy -g '!pythnINS' -g'!Python' -g '!fhirconverter' -g '!transformer' -g '!Downloads' $2
  fi
}

ntScrll()
{
  mouseID=`xinput list|awk '/Mouse.*id=/ {print $8}'|awk '{gsub("id=", ""); print $0}'`
  ntscrllProp=`xinput list-props $mouseID |awk -e '/libinput Natural Scrolling Enabled \(.*\):/ {gsub("[():]", "");print $5}'`
  xinput set-prop $mouseID $ntscrllProp 1
}

gtkQuery()
{
  gtk-query-immodules-2.0 >/tmp/gtk.immodules
  sudo mv /etc/gtk-2.0/gtk.immodules  /etc/gtk-2.0/ori.gtk.immodules
  sudo mv /tmp/gtk.immodules /etc/gtk-2.0/
}

lprRange()
{
lpr -o A4 -o fit-to-page -o media=A4 -o page-ranges=$1 $2
}

probeVid()
{
  duration=`ffprobe $1 2>&1|grep Duration|awk -F, '{print $1}'|awk '{print $2}'`
  echo $1 $duration
}

scpCancer()
{
scp -P 4822 cancer:$1 .
}

envLdLbrryPth(){
  realPython=`which python`
  #echo $realPython
  rlPythnPath=`readlink $realPython`
  #echo $rlPythnPath
  parentPath=$(dirname $(dirname "$rlPythnPath"))
  #echo $parentPath
  env LD_LIBRARY_PATH="$parentPath/lib" $1 $2
}

alias importScrn='import -window root -pause 3 -quality 90 /tmp/tmp.png'
#-crop 512x256-0-0 -gravity northeast 
PrntScrn()
{
tmpFile=/tmp/"$1".png
scrot -d8 -o $tmpFile
#import -window root -pause 8 $tmpFile
gimp $tmpFile&
}

TERMINAL_SESSIONS_DIR=~/.gnome/sssn #terminal-sessions
saveSSSN(){
gnome-terminal --save-config=$TERMINAL_SESSIONS_DIR/$(date +"%m%d%Y-%H%M%S")
}

loadSSSN(){
NEWEST_FILENAME=$(ls -t $TERMINAL_SESSIONS_DIR | head -1)
gnome-terminal --load-config=$TERMINAL_SESSIONS_DIR/$NEWEST_FILENAME
}

prntscrn()
{
tmpFile=/tmp/"$2".png
import -window root -pause $1 $tmpFile
gimp $tmpFile&
}
#prntscrn demo.png
#alias prntscrn='import -window root -pause 4 /tmp/demo.png'
Jupyter()
{
IP=`which ip`
IP=`${IP} addr show dev eno1`
IP=`echo $IP|awk '{match($0,/[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/); ip = substr($0,RSTART,RLENGTH); print ip}'`
jupyter notebook --ip $IP
}
#alias vdn="$HOME/devSys/neovide.AppImage"
