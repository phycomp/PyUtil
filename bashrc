source /etc/profile.d/vte.sh
# If not running interactively, don't do anything
[[ $- != *i* ]] && return

alias ls='ls --color=auto'
alias grep='grep --color=auto'

export XDG_CONFIG_HOME=$HOME/.config
. $HOME/.bash_aliases
#export pyVerLen=${#VIRTUAL_ENV_PROMPT}
#sudo swapon
PS1='[\u@\h \W]\$ '
ipUtil=`which ip`
IP=`${ipUtil} addr show dev eno1`
IP=`echo $IP|awk '{match($0,/[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/); ip = substr($0,RSTART,RLENGTH); print ip}'`
if [ -z $IP ]; then
  IP=10.121.12.128
  sudo ip link set eno1 up
  sudo ip addr add 10.121.12.128/16 dev eno1
  sudo ip route add default via 10.121.254.254
#else
  #jupyter notebook --ip $IP
fi
export PS1="[\u@Δ $IP \W]"
#export IFS=$'\n'
#export PS1='[\u@\h $IP \W]\$ '
#rgpy -g '!pythnINS' -g'!Python' -g '!python' -g '!fhirconverter' -g '!transformer' -g '!Downloads' -g '!*.html' -g '!*.js'  -g '!vtk' -g '!libre' -g '!go'
#export PATH="/home/josh/.cache/pypoetry/virtualenvs/medset-qcpfaqsP-py3.11/bin:/home/josh/chromium/chrome-linux:/home/josh/nvimINS/bin::/home/josh/pythnINS/bin:/home/josh/PyUtils:/home/josh/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/usr/bin/site_perl:/usr/bin/vendor_perl:/usr/bin/core_perl:/usr/lib/rustup/bin:/home/josh/.vimpkg/bin"
#export LD_LIBRARY_PATH =$LD_LIBRARY_PATH
LD_LIBRARY_PATH=$HOME/jdk-22/lib:/usr/lib:$HOME/sslINS/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/jre1.8.0_411/lib/amd64/jli:$LD_LIBRARY_PATH

PATH=$HOME/jdk-22/bin/:$HOME/rangerINS/bin:"/home/josh/.cache/pypoetry/virtualenvs/medset-qcpfaqsP-py3.11/bin:/home/josh/.cache/pypoetry/virtualenvs/medset-qcpfaqsP-py3.11/bin:/home/josh/chromium/chrome-linux:/home/josh/nvimINS/bin::/home/josh/pythnINS/bin:/home/josh/PyUtils:/home/josh/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/bin:/usr/bin/site_perl:/usr/bin/vendor_perl:/usr/bin/core_perl:/usr/lib/rustup/bin:/home/josh/.vimpkg/bin:/home/josh/.vimpkg/bin"
export PATH=$HOME/jre1.8.0_411/bin:$PATH
#$HOME/dbeaver/jre/bin/:
PROMPT_COMMAND='echo -ne "\033]0;Δ ${IP}\007"'
export PYTHONPATH=$HOME/devSys/bdcMNPL:$PYTHONPATH
export QT_SCALE_FACTOR=1
if [[ $DISPLAY ]]; then
  #xset q &>/dev/null -z $DISPLAY
  #echo "No X server at \$DISPLAY [$DISPLAY]" >&2
  #exit 1
  #xinput set-prop 8 281 1
  #xinput set-prop 11 281 1
  ntScrll
  #xinput set-prop 10 307 1
  sleep 1
  #$HOME/vghXmodmap
fi
export VISUAL=nvim
export EDITOR=nvim

# JINA_CLI_BEGIN

## autocomplete
_jina() {
  COMPREPLY=()
  local word="${COMP_WORDS[COMP_CWORD]}"

  if [ "$COMP_CWORD" -eq 1 ]; then
    COMPREPLY=( $(compgen -W "$(jina commands)" -- "$word") )
  else
    local words=("${COMP_WORDS[@]}")
    unset words[0]
    unset words[$COMP_CWORD]
    local completions=$(jina completions "${words[@]}")
    COMPREPLY=( $(compgen -W "$completions" -- "$word") )
  fi
}

complete -F _jina jina

# session-wise fix
ulimit -n 4096
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
# default workspace for Executors

# JINA_CLI_END

#sudo systemctl stop updatedb.timer

#[ -f ~/.fzf.bash ] && source ~/.fzf.bash
#export PYTHONPATH=$HOME/vwrbdc/bdcMNPL:$PYTHONPATH
#xauth add "$DISPLAY" MIT-MAGIC-COOKIE-1 <hex key>

# bun
export BUN_INSTALL="$HOME/.local/share/reflex/bun"
export PATH="$BUN_INSTALL/bin:$PATH"
#psql postgresql://postgres@10.221.252.51:5437/eyeVis
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

# >>> Added by Spyder >>>
#alias spyder=/home/josh/.local/spyder-6/envs/spyder-runtime/bin/spyder
#alias uninstall-spyder=/home/josh/.local/spyder-6/uninstall-spyder.sh
# <<< Added by Spyder <<<

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
#__conda_setup="$('/home/josh/.local/spyder-6/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/home/josh/.local/spyder-6/etc/profile.d/conda.sh" ]; then
#        . "/home/josh/.local/spyder-6/etc/profile.d/conda.sh"
#    else
#        export PATH="/home/josh/.local/spyder-6/bin:$PATH"
#    fi
#fi
#unset __conda_setup
# <<< conda initialize <<<
