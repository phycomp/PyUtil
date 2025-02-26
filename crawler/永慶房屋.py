import requests
from bs4 import BeautifulSoup
import pandas as pd
from fake_useragent import UserAgent
import time
import json
from datetime import datetime
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class YungChingCrawler:
    def __init__(self):
        self.ua = UserAgent()
        self.headers = {
            'User-Agent': self.ua.random
        }
        
    def get_city_urls(self):
        """獲取各城市的永慶房屋網址"""
        return {
            '台北市': 'https://buy.yungching.com.tw/region/%E5%8F%B0%E5%8C%97%E5%B8%82/_hw',
            '新北市': 'https://buy.yungching.com.tw/region/%E6%96%B0%E5%8C%97%E5%B8%82/_hw',
            '桃園市': 'https://buy.yungching.com.tw/region/%E6%A1%83%E5%9C%92%E5%B8%82/_hw',
            '台中市': 'https://buy.yungching.com.tw/region/%E5%8F%B0%E4%B8%AD%E5%B8%82/_hw',
            '台南市': 'https://buy.yungching.com.tw/region/%E5%8F%B0%E5%8D%97%E5%B8%82/_hw',
            '高雄市': 'https://buy.yungching.com.tw/region/%E9%AB%98%E9%9B%84%E5%B8%82/_hw'
        }

    def parse_house_data(self, html, city):
        """解析房屋資料"""
        soup = BeautifulSoup(html, 'html.parser')
        houses = []
        
        for item in soup.select('.m-list-item'):
            try:
                price_text = item.select_one('.price').text.strip()
                price = float(re.findall(r'\d+\.?\d*', price_text)[0])
                
                area_text = item.select_one('.area').text.strip()
                area = float(re.findall(r'\d+\.?\d*', area_text)[0])
                
                house_info = {
                    '標題': item.select_one('.title').text.strip(),
                    '總價': price,
                    '建物面積': area,
                    '單價': round(price / area, 2),
                    '建物類型': item.select_one('.item-style').text.strip(),
                    '區域': item.select_one('.item-area').text.strip(),
                    '地址': item.select_one('.address').text.strip(),
                    '來源': '永慶房屋',
                    '縣市': city
                }
                houses.append(house_info)
            except Exception as e:
                print(f"解析錯誤: {e}")
                continue
                
        return houses

    def crawl_city_data(self, city, url, max_pages=3):
        """爬取特定城市的房屋資料"""
        houses = []
        
        for page in range(1, max_pages + 1):
            page_url = f"{url}?pg={page}"
            try:
                response = requests.get(page_url, headers=self.headers)
                if response.status_code == 200:
                    houses.extend(self.parse_house_data(response.text, city))
                time.sleep(2)
            except Exception as e:
                print(f"爬取錯誤: {e}")
                continue
                
        return houses

class SinYiCrawler:
    def __init__(self):
        self.options = Options()
        self.options.add_argument('--headless')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=self.options)
        
    def get_city_urls(self):
        """獲取各城市的信義房屋網址"""
        return {
            '台北市': 'https://www.sinyi.com.tw/buy/list/taipei-city/residential',
            '新北市': 'https://www.sinyi.com.tw/buy/list/new-taipei-city/residential',
            '桃園市': 'https://www.sinyi.com.tw/buy/list/taoyuan-city/residential',
            '台中市': 'https://www.sinyi.com.tw/buy/list/taichung-city/residential',
            '台南市': 'https://www.sinyi.com.tw/buy/list/tainan-city/residential',
            '高雄市': 'https://www.sinyi.com.tw/buy/list/kaohsiung-city/residential'
        }
        
    def parse_house_data(self, city):
        """解析房屋資料"""
        houses = []
        wait = WebDriverWait(self.driver, 10)
        items = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.buy-list-item')))
        
        for item in items:
            try:
                price_text = item.find_element(By.CSS_SELECTOR, '.price').text.strip()
                price = float(re.findall(r'\d+\.?\d*', price_text)[0])
                
                area_text = item.find_element(By.CSS_SELECTOR, '.area').text.strip()
                area = float(re.findall(r'\d+\.?\d*', area_text)[0])
                
                house_info = {
                    '標題': item.find_element(By.CSS_SELECTOR, '.title').text.strip(),
                    '總價': price,
                    '建物面積': area,
                    '單價': round(price / area, 2),
                    '建物類型': item.find_element(By.CSS_SELECTOR, '.item-type').text.strip(),
                    '區域': item.find_element(By.CSS_SELECTOR, '.item-area').text.strip(),
                    '地址': item.find_element(By.CSS_SELECTOR, '.address').text.strip(),
                    '來源': '信義房屋',
                    '縣市': city
                }
                houses.append(house_info)
            except Exception as e:
                print(f"解析錯誤: {e}")
                continue
                
        return houses

    def crawl_city_data(self, city, url, max_pages=3):
        """爬取特定城市的房屋資料"""
        houses = []
        
        try:
            for page in range(1, max_pages + 1):
                page_url = f"{url}?pg={page}"
                self.driver.get(page_url)
                houses.extend(self.parse_house_data(city))
                time.sleep(2)
        except Exception as e:
            print(f"爬取錯誤: {e}")
        finally:
            self.driver.quit()
            
        return houses

def get_all_house_data():
    """獲取所有來源的房屋資料"""
    # 建立爬蟲實例
    yungching_crawler = YungChingCrawler()
    sinyi_crawler = SinYiCrawler()
    
    all_data = []
    
    # 爬取永慶房屋資料
    for city, url in yungching_crawler.get_city_urls().items():
        print(f"正在爬取永慶房屋 {city} 的資料...")
        houses = yungching_crawler.crawl_city_data(city, url)
        all_data.extend(houses)
        time.sleep(3)
    
    # 爬取信義房屋資料
    for city, url in sinyi_crawler.get_city_urls().items():
        print(f"正在爬取信義房屋 {city} 的資料...")
        houses = sinyi_crawler.crawl_city_data(city, url)
        all_data.extend(houses)
        time.sleep(3)
    
    # 轉換為DataFrame並保存
    if all_data:
        df = pd.DataFrame(all_data)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'house_data_{timestamp}.csv'
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        return df
    return pd.DataFrame()
