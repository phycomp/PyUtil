from pytrends.request import TrendReq
from stUtil import rndrCode
from datetime import datetime, timedelta

趨勢 = TrendReq(hl='zh-TW', tz=360) # 設定 Google 趨勢的請求

today = datetime.today() # 設定要取得的時間範圍
yesterday = today - timedelta(days=1)
timeframe = f'{yesterday.strftime("%Y-%m-%d")} {today.strftime("%Y-%m-%d")}'
PN='taiwan'

每日趨勢=趨勢.trending_searches(pn=PN) # 取得台灣的每日熱門關鍵詞
rndrCode(["每日趨勢搜尋：", 每日趨勢])
srchTrend=趨勢.trending_searches(pn=PN)   #'Country'
實時趨勢=趨勢.realtime_trending_searches(pn=PN)
#實時趨勢=趨勢.realtime_trending_searches(pn='Taiwan')  #'Country' 
rndrCode(["實時趨勢搜尋：", srchTrend, 實時趨勢]) #實時趨勢.head(10)

#實時趨勢 = 趨勢.realtime_trending_searches(pn='TW') # 取得台灣的實時熱門關鍵詞
#Pochi發佈於程式輕鬆玩 https://vocus.cc/article/664e242afd89780001360191
