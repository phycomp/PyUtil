#!/usr/bin/env python
from pathlib import Path
from os import system
from sys import argv

文件=Path(argv[1])
匯出=文件.with_suffix('.pdf')
#文件.rename(文件+'.pdf')
命令=f'pandoc {文件} -o {匯出}'
命令+=r""" -V colorlinks=true -V linkcolor=blue -V emojifont='Segoe UI Emoji' -V emojifont='EmojiOne Color' -V urlcolor=red -V toccolor=gray --pdf-engine=xelatex -V CJKmainfont='Noto Serif CJK SC' -V CJKmonofont='Noto Serif CJK SC' -V mainfont='DejaVu Sans' -V monofont='DejaVu Sans' -V geometry:top=1.2cm,bottom=1.2cm,left=1.2cm,right=1.2cm  --syntax-highlighting=tango"""
#-V header-includes='\usepackage{fontspec}\newfontfamily\emojifont{Noto Color Emoji}[Renderer=HarfBuzz,RawFeature={mode=harf}]\DeclareTextFontCommand{\emoji}{\emojifont}'"""

#print(命令)
system(命令)
