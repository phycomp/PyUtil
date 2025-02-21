class CaptchaMiddleware(object):
    max_retries = 5
    def process_response(request, response, spider):
        if not request.meta.get('solve_captcha', False):
            return response  # only solve requests that are marked with meta key
        catpcha = find_catpcha(response)
        if not captcha:  # it might not have captcha at all!
            return response
        solved = solve_captcha(captcha)
        if solved:
            response.meta['catpcha'] = captcha
            response.meta['solved_catpcha'] = solved
            return response
        else:
            # retry page for new captcha
            # prevent endless loop
            if request.meta.get('catpcha_retries', 0) == 5:
                logging.warning('max retries for captcha reached for {}'.format(request.url))
                raise IgnoreRequest 
            request.meta['dont_filter'] = True
            request.meta['captcha_retries'] = request.meta.get('captcha_retries', 0) + 1
            return request
This example will intercept every response and try to solve the captcha. If failed it will retry the page for new captcha, if successful it will add some meta keys to response with solved captcha values.
In your spider you would use it like this:

class MySpider(scrapy.Spider):
    def parse(self, response):
        url = ''# url that requires captcha
        yield Request(url, callback=self.parse_captchad, meta={'solve_captcha': True},
                      errback=self.parse_fail)

    def parse_captchad(self, response):
        solved = response['solved']
        # do stuff

    def parse_fail(self, response):
        # failed to retrieve captcha in 5 tries :(
        # do stuff


from time import sleep
from scrapy import Spider, Request
from scrapy.http import FormRequest
from io import BytesIO
from re import search, compile, IGNORECASE
from PIL.Image import open as img_open
from tesserocr import image_to_text
from lxml.etree import fromstring, HTMLParser
users=['W-AB99667', 'lu5796a', 'lu5796b', 'lu5796c', 'lu5796d', 'lu5796e', 'lu5796f', 'lu5796g', 'lu5796h', 'lu5796i', 'lu5796j', 'lu5796k', 'lu5796l', 'lu5796m', 'lu5796n', 'lu5796o', 'lu5796p', 'lu5796q', 'lu5796r', 'lu5796s', 'lu5796t', 'lu5796u', 'lu5796v', 'lu5796w', 'lu5796x', 'lu5796y', 'lu5796z', 'lu5796aa']

class SkyRtrv(Spider):
	hParser=HTMLParser()
	name = 'sky-rtrv'
	allowed_domains = ['sky2628.com']
	login_url = r'https://www.sky2628.com/System/Security/HYLogin.aspx'
	captcha_url = r'https://www.sky2628.com/System/Security/Turing.aspx'
	eMarket_url=r'https://www.sky2628.com/System/Share/Reports/eMarketStatement.aspx'
	resizeRatio=3
	dateFlag=False
	start_urls = [login_url]
	headers={'Accept-Language':'en'}
	spanID='//span/@id'
	username=compile('username', IGNORECASE)
	WalletID=compile('WalletID', IGNORECASE)
	walletID=compile('fa fa-user fa-fw')
	totalSplits=compile('ctl00_ContentPlaceHolder1_lblTotalSplitTimes')
	credit=compile('ctl00_ContentPlaceHolder1_lblCredit')
	debit=compile('ctl00_ContentPlaceHolder1_lblDebit')
	balance=compile('ctl00_ContentPlaceHolder1_lblBalance')
	CreateDate=compile('lblCreateDate')
	def parse(self, response):
		yield Request(url=self.captcha_url, callback=self.rtrv_captcha, meta={'previous_response':response})
		#yield Request(url=self.login_url, callback=self.login)	#, meta={'previous_response':response})
	def rtrv_captcha(self, response):
		rBody=BytesIO(response.body)
		im=img_open(rBody)
		nimSize=map(lambda value: self.resizeRatio*value, im.size)
		nim=im.resize(nimSize)
		self.txtCAPTCHA=image_to_text(nim.convert('1')).strip('\n').strip(' ')
		print('txtCAPTCHA=', self.txtCAPTCHA)
		self.formdata=dict(txtUserName='W-AB99667', txtPassword='Taote-2588', txtCAPTCHA=self.txtCAPTCHA, cboLanguage='en-US')
		return FormRequest.from_response(response=response.meta['previous_response'], url=self.login_url, formdata=self.formdata, callback=self.after_login)
	def after_login(self, response):
		if b"authentication failed" in response.body:
			self.logger.error("Login failed")
			return
		sleep(4)
		return Request(url=self.eMarket_url, callback=self.rtrv_emarket)

	def rtrv_emarket(self, response):
		rBody=response.body
		self.Tree=fromstring(rBody, self.hParser)
		self.treeXpath=self.Tree.xpath(self.spanID)
		for pttrn in self.treeXpath:
			attrSpanID='//span[@id="%s"]'%pttrn
			e=self.Tree.xpath(attrSpanID)[0]
			info=e.text
			if info:
				if self.username.search(pttrn): print('%s'%info, end=', ')
				elif self.WalletID.search(pttrn): print(info, end=', ')
				elif self.walletID.search(pttrn): print(info, end=', ')
				elif self.totalSplits.search(pttrn): print('%s'%info, end=', ')
		print()
		for pttrn in self.treeXpath:
			attrSpanID='//span[@id="%s"]'%pttrn	#'//td//span[@id="ctl00_ContentPlaceHolder1_dgList_ctl09_lblBalance"]'
			e=self.Tree.xpath(attrSpanID)[0]
			info=e.text
			if info:
				if self.credit.search(pttrn) and not search('Limit', pttrn): print('%s'%info, end=', ')
				elif self.debit.search(pttrn): print('%s'%info, end=', ')
				elif self.balance.search(pttrn): print('%s'%info)
				elif self.CreateDate.search(pttrn):
					print('\n%s'%info, end=', ')
					if not self.dateFlag: self.dateFlag=True
				elif self.dateFlag: print(info, end=', ')
'''
		#emarketform={'ctl00$ContentPlaceHolder1$txtFrom':'2018/03/01', '__VIEWSTATEGENERATOR':'6B892676', 'ctl00$ContentPlaceHolder1$btnSubmit':'View'}
		#return FormRequest.from_response(response=response, url=self.eMarket_url, formdata=emarketform, callback=self.rtrv_emarket)
#<input type="submit" name="ctl00$ContentPlaceHolder1$btnSubmit" value="View" onclick="return checkConfirm('S', 'ctl00_ContentPlaceHolder1_txtWalletID', 'ctl00_ContentPlaceHolder1_txtFrom', '', 'en-US');" id="ctl00_ContentPlaceHolder1_btnSubmit" class="btn btn-primary">
	def login(self , response):
		yield FormRequest(url=self.login_url, formdata=self.data, callback=self.parse)
yield Request(url, callback=self.parse, method="POST", body=urllib.urlencode(frmdata))
def parse(self, response):
	open_in_browser(response)
	links = response.xpath('//p/a/@href').extract()
	for link in links:
		absoulute_url = response.urljoin(link)
	yield Request(self.captcha_url, callback=self.rtrv_captcha)
#from scrapy.utils.response import open_in_browser
from urllib3.util.url import parse_url
from argparse import ArgumentParser
		with open('captcha.png', 'wb') as f:
			f.write(response.body)
		captcha = raw_input("-----> Enter the captcha in manually :")

		return FormRequest.from_response(
		response=response.meta['previous_response'],
		"captcha_code": captcha},
		formxpath="//*[@id='login-form']",
		callback=self.after_login)
	def __init__(self, args):
		self.username, self.password=args.username, args.password
		self.data = dict(txtUserName=self.username, txtPassword=self.password, txtCAPTCHA=self.captcha)
if __name__=='__main__':
	parser = ArgumentParser(description='calculate stock to the total of SKY')
	parser.add_argument('--password', '-p', default='Taote-2588', help='the password')
	parser.add_argument('--username', '-u', default='W-AB99667', help='the username')
	parser.add_argument('--captcha', '-c', help='the captcha')
	parser.add_argument('--Login', '-L', action='store_true', help='the Share stock information')
	args = parser.parse_args()
	if args.Login:
		skyspider=SkySpider(args)
		skyspider.login()
'''
