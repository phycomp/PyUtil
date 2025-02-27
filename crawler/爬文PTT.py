#!/usr/bin/env python
#from urllib import urlparse
import requests

def fetch(url):
    response = requests.get(url)
    response = requests.get(url, cookies={'over18': '1'})  # 一直向 server 回答滿 18 歲了 !
    return response

url='https://www.ptt.cc/bbs/CodeJob/M.1540970567.A.B09.html'
#url = 'https://www.ptt.cc/bbs/movie/index.html'
resp = fetch(url)  # step-1

INFOs=["article-metaline", "article-metaline", "article-metaline-right", "article-metaline", "article-meta-value"]
#<div id="main-content" class="bbs-screen bbs-content">
#<div class="article-metaline"><span class="article-meta-tag">作者</span><span class="article-meta-value">taipei49314 (嘟嘟嚕嚕嚕嚕)</span></div>
#<div class="article-metaline-right"><span class="article-meta-tag">看板</span><span class="article-meta-value">CodeJob</span></div>
#<div class="article-metaline"><span class="article-meta-tag">標題</span><span class="article-meta-value">[發案] matlab病情分析</span></div>
#<div class="article-metaline"><span class="article-meta-tag">時間</span><span class="article-meta-value">Wed Oct 31 15:22:44 2018</span></div>
rBody=open('chu.html').read()	#resp.text # result of setp-1
from lxml.etree import fromstring, HTMLParser
hParser=HTMLParser()
Tree=fromstring(rBody, hParser)
spanClass='//div/@class'
for pttrn in INFOs:
	#Tree.xpath(spanClass):
	attrSpanClass='//span[@class="%s"]'%pttrn
	rslt=Tree.xpath(attrSpanClass)[0]
	info=rslt.text
	if info:
		print(info)
	#Tree.xpath(attrSpanID)[0]
#class="article-meta-tag">
#• 日期
#• 作者
#• 標題
#• 內文
#• 看板名稱
