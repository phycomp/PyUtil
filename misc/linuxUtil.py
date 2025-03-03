#!/usr/bin/bash replaceHOSTNAME.sh 
export hstname=`awk -F '=' '/^HOSTNAME=/{print $2}' .ORI.bashrc` && sed -i "/^HOSTNAME=/c HOSTNAME=$hstname" .bashrc

#!/usr/bin/bash sed2pynvim.sh 
echo $VIRTUAL_ENV
sed -ie 's/import imp/import importlib/g' $VIRTUAL_ENV/lib/python3.12/site-packages/pynvim/plugin/script_host.py
sed -ie 's/import imp/import importlib/g' $VIRTUAL_ENV/lib/python3.12/site-packages/pynvim/plugin/host.py
sed -ie 's/from imp import find_module/from importlib import import_moduel/g' $VIRTUAL_ENV/lib/python3.12/site-packages/pynvim/compat.py

#!/usr/bin/bash
#dbeaverUtil.sh -->新增dbeaver版本 使用ar將.deb解開
rm -rf ~/Downloads/dbeaver-ce/*
cd ~/Downloads
ar x dbeaver-ce*.deb data.tar.gz
tar zxvf data.tar.gz -C ~/Downloads/dbeaver-ce/
mv dbeaver-ce*.deb ~/Downloads/dbeaver-ce/
rm -rf ~/ori.dbeaver
mv ~/dbeaver ~/ori.dbeaver
mv ~/Downloads/dbeaver-ce/usr/share/dbeaver-ce/ ~/dbeaver
