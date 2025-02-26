import requests
import json
import time
from pandas import DataFrame
from fake_useragent import UserAgent
from datetime import datetime, timedelta

class House591Crawler:
    def __init__(self):
        self.ua = UserAgent()
        self.headers = {
            'User-Agent': self.ua.random,
            'X-CSRF-TOKEN': '',
        }
        self.session = requests.Session()
        self._get_csrf_token()

    def _get_csrf_token(self):
        """獲取CSRF Token"""
        url = 'https://sale.591.com.tw/'
        response = self.session.get(url, headers=self.headers)
        csrf_token = response.cookies.get('csrf_token')
        self.headers['X-CSRF-TOKEN'] = csrf_token

    def get_region_ids(self):
        """獲取地區ID對應表"""
        region_map = {
            '台北市': 1,
            '新北市': 3,
            '桃園市': 6,
            '台中市': 8,
            '台南市': 15,
            '高雄市': 17
        }
        return region_map

    def search_houses(self, region_id, page=1):
        """搜尋特定地區的房屋資料"""
        url = 'https://sale.591.com.tw/home/search/list'
        params = {
            'region': region_id,
            'page': page,
            'type': 2,  # 2為屋單
            'sort': 'price_asc',
            'shType': 'list'
        }

        try:
            response = self.session.get(url, params=params, headers=self.headers)
            data = response.json()
            return data
        except Exception as e:
            print(f"錯誤: {e}")
            return None

    def parse_house_data(self, raw_data):
        """解析房屋資料"""
        if not raw_data or 'data' not in raw_data or 'house_list' not in raw_data['data']:
            return DataFrame()

        houses = []
        for house in raw_data['data']['house_list']:
            try:
                house_info = {
                    '標題': house.get('title', ''),
                    '總價': float(house.get('price', '0').replace('萬', '')),
                    '建物面積': float(house.get('area', '0')),
                    '單價': float(house.get('unit_price', '0').replace('萬/坪', '')),
                    '建物類型': house.get('kind_name', ''),
                    '區域': house.get('region_name', ''),
                    '地址': house.get('address', ''),
                    '樓層': house.get('floor_str', ''),
                    '建築年份': house.get('build_year', ''),
                    '更新時間': house.get('refresh_time', ''),
                    '房屋ID': house.get('houseid', '')
                }
                houses.append(house_info)
            except Exception as e:
                print(f"解析錯誤: {e}")
                continue

        return DataFrame(houses)

    def crawl_region_data(self, region_id, max_pages=3):
        """爬取特定地區的所有房屋資料"""
        all_data = []

        for page in range(1, max_pages + 1):
            print(f"正在爬取第 {page} 頁...")
            raw_data = self.search_houses(region_id, page)
            if raw_data:
                df = self.parse_house_data(raw_data)
                if not df.empty:
                    all_data.append(df)
                else:
                    break
            time.sleep(2)  # 避免訪問過快

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return DataFrame()

def get_591_data():
    """獲取所有地區的591資料"""
    crawler = House591Crawler()
    region_ids = crawler.get_region_ids()
    all_data = []

    for region_name, region_id in region_ids.items():
        print(f"正在爬取{region_name}的資料...")
        df = crawler.crawl_region_data(region_id)
        if not df.empty:
            df['縣市'] = region_name
            all_data.append(df)
        time.sleep(3)  # 區域之間的延遲

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # 保存到CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'591_house_data_{timestamp}.csv'
        final_df.to_csv(filename, index=False, encoding='utf-8-sig')
        return final_df
    return DataFrame()
