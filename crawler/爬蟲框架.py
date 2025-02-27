import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import time
import random
import csv
import json


class Crawler:
    def __init__(self, start_urls, max_workers=5, retry=3, delay=2):
        """
        初始化爬蟲框架。
        :param start_urls: 初始抓取的URL列表
        :param max_workers: 最大并發線程數
        :param retry: 請求失敗時的重試次數
        :param delay: 每次請求之間的延遲時間，防止服務器壓力過大
        """
        self.start_urls = start_urls
        self.max_workers = max_workers
        self.retry = retry
        self.delay = delay
        self.crawled_data = []
    
    def fetch(self, url, retries=0):
        """
        處理HTTP請求并獲取頁面內容。
        :param url: 請求的URL
        :param retries: 重試次數
        :return: 返回HTML內容或None
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            time.sleep(self.delay + random.random())  # 加入隨機延遲，防止被封
            return response.text
        except requests.RequestException as e:
            if retries < self.retry:
                print(f"Retrying {url} ({retries+1}/{self.retry})")
                return self.fetch(url, retries+1)
            else:
                print(f"Failed to fetch {url} after {self.retry} retries: {e}")
                return None
    
    def parse(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title').get_text()
        return {'title': title}
    
    def save_data(self, data, output_format='csv', output_file='output.csv'):
        """
        保存抓取到的數據到本地文件。
        :param data: 要保存的數據
        :param output_format: 文件格式，支持csv或json
        :param output_file: 保存的文件名
        """
        if output_format == 'csv':
            with open(output_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        elif output_format == 'json':
            with open(output_file, mode='w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
    
    def crawl(self, url):
        """
        對單個URL進行抓取和解析。
        :param url: 要抓取的URL
        """
        print(f"Crawling: {url}")
        html = self.fetch(url)
        if html:
            data = self.parse(html)
            if data:
                self.crawled_data.append(data)
    
    def run(self):
        """
        啓動爬蟲，處理所有URL。
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(self.crawl, self.start_urls)
        # 抓取完成後保存數據
        self.save_data(self.crawled_data)


if __name__ == "__main__":
    start_urls = [
        "https://example.com", 
        "https://example.org", 
        "https://example.net"
    ]
    
    crawler = Crawler(start_urls)
    crawler.run()
