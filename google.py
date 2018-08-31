#!/usr/bin/env python
from urllib3 import PoolManager
from os import system
from sys import argv
from re import findall

http = PoolManager()
rqst = http.request('GET', argv[1])	#'https://www.youtube.com/watch?v=B-oqeIEYnFc'
Data=rqst.data
Data=Data.decode('utf-8')
videos=findall(r'watch\?v=.{11}', Data)
videos=set(videos)
for vid in videos:
	cmd='cclive https://www.youtube.com/%s'%vid
	print(cmd)
	system(cmd)
