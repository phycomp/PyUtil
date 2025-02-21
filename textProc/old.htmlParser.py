#!/usr/bin/env python
#-*- coding=utf-8 -*-

from html.parser import HTMLParser
from glob import glob
from re import sub
#from htmlentitydefs import name2codepoint
from sys import argv

class _DeHTMLParser(HTMLParser):  
    def __init__(self):  
        HTMLParser.__init__(self)  
        self.__text = []  
    def handle_data(self, data):  
        text = data.strip()  
        if len(text) > 0:  
            text = sub('[ \t\r\n]+', ' ', text)  
            self.__text.append(text + ' ')  
    def handle_starttag(self, tag, attrs):  
        if tag == 'p':  self.__text.append('\n\n')  
        elif tag == 'br':  self.__text.append('\n')  
    def handle_startendtag(self, tag, attrs):  
        if tag == 'br':  self.__text.append('\n\n')  
    def text(self):  
        return ''.join(self.__text).strip()  
fout=open('merge', 'w')
Files=glob('*.html')
Files.sort()
for fname in Files:
	myParser=_DeHTMLParser()
	#print(fname)
	#Data=open(fname, errors='ignore').read()
	Data=open(fname, encoding='gb2312', errors='ignore').read()
	#Data=Data.encode('big5')
	myParser.feed(Data)
	#print(''.join(myParser.__text).strip())
	#print(myParser.text()+'\n')
	out=myParser.text()+'\n'
	fout.write(out)
	#myParser.__text=[]
fout.close()

'''
if __name__=='__main__':
	parser = ArgumentParser(description='html parser')
	parser.add_argument('--enc', '-e', action='store', default='big5', help='the default encoding')
	parser.add_argument('--Iter', '-I', action='store_true', default=False, help='iteration of gamma')
	args = parser.parse_args()
	if args.enc: calc_stock(args)

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('<%s>' % tag)
    def handle_endtag(self, tag):
        print('</%s>' % tag)
    def handle_startendtag(self, tag, attrs):
        print('<%s/>' % tag)
    def handle_data(self, data):
        print(data)
    def handle_comment(self, data):
        print('<!-- -->')
    def handle_entityref(self, name):
        print('&%s;' % name)
    def handle_charref(self, name):
        print('&#%s;' % name)
'''
