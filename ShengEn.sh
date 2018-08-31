#!/bin/bash
PWD=`pwd`
echo $PWD
wpa_supplicant -i wlp3s0 -c $PWD/ShengEn.conf &
dhcpcd wlp3s0
