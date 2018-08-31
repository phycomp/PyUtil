#!/usr/bin/env python

from os.path import dirname
from os import system
from sys import argv

conf, PWD=argv[1], dirname(__file__)
cmd='wpa_supplicant -i wlp3s0 -c %s&'%conf
print(cmd)
system(cmd)
#cmd='dhcpcd wlp3s0'
cmd='dhclient wlp3s0'
print(cmd)
system(cmd)
#wpa_passphrase A2-202 Luo@557888373932
