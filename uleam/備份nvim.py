#!/usr/bin/env python
from pathlib import Path
from os import system, chdir
備份目錄=Path('~/.config/nvim').expanduser()
目標檔案=Path('~/joshnvm.zip').expanduser()
# chdir(備份目錄)
# 命令='zip -r lua py sssn vim rplugin init.vim'
命令=f'zip -r {目標檔案} {備份目錄}/{{lua,py,vim,rplugin,init.vim}}'
#執行
print(命令)
system(命令)
# system(f'mv {備份目錄/joshnvm.zip} .')
