#!/usr/bin/bash
sudo yum install epel-release -y
sudo yum install centos-release-scl -y
sudo yum install devtoolset-8 -y
sudo yum install -y libzstd-devel
#0devtoolINS.sh --> 启用devtoolset-8
#scl enable devtoolset-8 -- bash
#source /opt/rh/devtoolset-8/enable
source scl_source enable devtoolset-8

#!/usr/bin/bash
# 1sysINS.sh-->
sudo yum install -y libffi-devel openssl-devel gcc g++ perl-Data-Dumper perl-IPC-Cmd libffi-devel wget xz-devel sqlite-devel bzip2-devel tcl-devel.x86_64 tk-devel.x86_64 vte-profile.x86_64 bash-completion readline-devel ncurses-devel tmux git libtool xsel

#!/usr/bin/bash
if [ -z $1 ]; then echo 'no opensslSRC' && exit; fi
export SSLSRC=$1  #openssl-1.1.1g.tar.gz
#2sslINS.sh -->tar zxvf $SSLSRC
export SSLDIR=`basename $SSLSRC .tar.gz`    #openssl-1.1.1g
export SSLINS=$HOME/sslINS
#rm -rf $SSLINS/*
mv $SSLINS ori.$SSLINS
cd $SSLDIR
make distclean
./config --prefix=$SSLINS
make && make install

#!/usr/bin/bash
#3pythonINS.sh-->安裝Python
export pythnSRC=$1  #`ls Python-3.*.xz`
#FILE=`basename $pythnSRC .tar.xz`
#basename  Python-3.12.0.tar.xz
export pythnDIR=`basename $pythnSRC .tar.xz`
if [ -z $1 ]; then echo 'no PythonSRC'&&exit; fi
if [ -d "${pythnDIR}" ] ; then
    echo "$pythnDIR is a directory";
else
  #if [ -f "${PASSED}" ]; then
  tar Jxvf $pythnSRC;
fi

export sslLIBDIR=`ls -d $HOME/sslINS/lib*`

export LD_LIBRARY_PATH=$sslLIBDIR:$LD_LIBRARY_PATH
export pythnPrefix=$HOME/pythnINS
rm -rf $pythnPrefix.ori/*
mv $pythnPrefix $pythnPrefix.ori/
cd $pythnDIR
#env LDFLAGS=-L$sslLIBDIR ./configure --prefix=$pythnPrefix --with-pkg-config=yes --with-openssl=$HOME/sslINS --with-ssl-default-suites=openssl
#make distclean
env CPPFLAGS="-I/home/josh/sqlINS/include" LDFLAGS="-L/home/josh/sqlINS/lib -L$sslLIBDIR" ./configure -C --with-openssl=$HOME/sslINS --prefix=$HOME/pythnINS --enable-optimizations --enable-loadable-sqlite-extensions #--with-openssl-rpath=auto OPENSSL_LDFLAGS=-L/home/josh/sslINS/lib64

#./configure --enable-loadable-sqlite-extensions
#--with-system-ffi
make PROFILE_TASK="-m test.regrtest --pgo disable"
make install
ln -s $pythnPrefix/bin/python3 $pythnPrefix/bin/python
#先安裝 sqlite3 自己編譯後 只需要將LD_LIBRARY_PATH 設定為 export LD_LIBRARY_PATH=/home/josh/sqlINS/lib:$LD_LIBRARY_PATH 指向sqlINS/lib 設定環境變數後 就可以使用sqlite3

(ljbeauty-py3.13) [josh@Δ 10.121.12.128 icdNLPOK]cat ~/devSys/
#!/usr/bin/bash
if [ -z $1 ]; then echo 'no cmakeSRC'&&exit; fi
export CMAEKSRC=$1
source scl_source enable devtoolset-8
tar zxvf $CMAEKSRC #cmake-3.24.2.tar.gz
export CMAKEDIR=`basename $CMAEKSRC .tar.gz`
cd $CMAKEDIR  #5cmakeINS.sh --> cmake-3.24.2
export CMAKEINS=$HOME/cmakeINS
./bootstrap --prefix=$CMAKEINS
make && make install

#/usr/bin/bash
export nvimSRC=$1
#export nvimDIR=$HOME/GITs/neovim
if [ -z $1 ] ; then echo "no $nvimSRC existed"&&exit ;
else
  echo '******* NVIM processing... ******* '
  #mkdir -p $GITs;
  #cd $HOME/GITs
  #rm -rf ori.neovim
  #mv neovim ori.neovim
  #git clone https://github.com/neovim/neovim
  #echo 'git clone'  
  if [ -d "$nvimSRC" ]; then
    #echo "$nvimSRC directory";
  #else
    #export nvimDIR=$HOME/neovim-nightly #`basename $nvimSRC .tar.gz`
    cmd="tar zxvf $nvimSRC -C $HOME"	 #nvimDIR
    #6nvimINS.sh  -->  安裝nvimSRC
    echo "cd $nvimSRC"
    cd $nvimSRC
    #echo `pwd`
    #source scl_source enable devtoolset-8
    #export CPATH=/opt/rh/devtoolset-8/root/usr/lib/gcc/x86_64-redhat-linux/8/include
  fi
fi

export PATH=$HOME/cmakeINS/bin:$PATH
export nvimINS=$HOME/nvimINS
rm -rf $HOME/ori.nvimINS/*
mv $HOME/nvimINS $HOME/ori.nvimINS
#rm -rf $nvimINS/*
#make distclean
make CMAKE_BUILD_TYPE=RelWithDebInfo CMAKE_INSTALL_PREFIX=$nvimINS && make install

#!/usr/bin/bash
#9paqINS.sh 安裝paq-nvim --> lua PAQ.lua init.lua window.lua settings.lua buffer.lua
sudo yum install -y xsel
git clone --depth=1 https://github.com/savq/paq-nvim.git "${XDG_DATA_HOME:-$HOME/.local/share}"/nvim/site/pack/paqs/start/paq-nvim
mv -f PAQ.lua $HOME/.local/share/nvim/site/pack/paqs/start/paq-nvim/lua/paq.lua
rm -rf $HOME/.config/nvim/lua
mv -f lua $HOME/.config/nvim/
mv -f init.lua $HOME/.config/nvim/

#!/usr/bin/bash
sudo yum install -y bison
export glibcSRC=$1  #`ls Python-3.*.xz`
if [ -z $1 ]; then echo 'no glibcSRC'&&exit; fi
tar Jxvf $glibcSRC
#10glibcINS.sh  --> ../glibc-2.36/configure --prefix=$HOME/glibcINS
export glibcDIR=`basename $glibcSRC .tar.xz`
export LD_LIBRARY_PATH=..
cd $glibcDIR
echo `pwd`
if [ -d build ]; then cd build; else mkdir build && cd build; fi
../configure --prefix=$HOME/glibcINS
make && make install
