from pytrends.request import TrendReq
from stUtil import rndrCode

pytrends = TrendReq(hl='en-US', tz=360) # 初始化 Pytrends

keywords = ["iPhone", "MacBook"] # 搜索的關鍵字

pytrends.build_payload(kw_list=keywords) # 獲取趨勢數據
#3. 分析趨勢數據 我們可以通過 Pytrends 獲取不同維度的趨勢數據，比如興趣隨時間的變化、不同地區的搜索趨勢、相關查詢等。
#(1) 獲取趨勢隨時間變化的數據 廣告主可以根據趨勢變化選擇最佳的廣告投放時機，針對不同時間段的搜索量高峰進行廣告推送。

# 獲取興趣隨時間的變化
interest_over_time_df = pytrends.interest_over_time()
rndrCode(interest_over_time_df.head())
#(2) 地區分析 廣告主可以通過分析不同地區的搜索興趣來定位廣告，將廣告投放到特定地區以提高轉化率。

# 獲取興趣按地區分布
interest_by_region_df = pytrends.interest_by_region()
rndrCode(interest_by_region_df.head())
#(3) 相關查詢 相關查詢可以幫助發現與廣告主產品相關的其他關鍵字，擴展廣告投放的關鍵字覆蓋范圍。

# 獲取相關查詢數據
related_queries_dict = pytrends.related_queries()
rndrCode(related_queries_dict)
#(4) 上升趨勢的搜索詞 廣告主可以根據最新的上升趨勢詞匯，投放相關廣告，以搶占市場的熱點。

# 獲取上升趨勢的搜索詞
rising_searches_df = pytrends.trending_searches()
rndrCode(rising_searches_df.head())
#4. 精准廣告投放的實踐 將 Pytrends 提供的數據應用於廣告投放，可以針對以下几方面進行優化：
#關鍵字投放優化：根據 Pytrends 提供的熱門搜索詞和相關搜索詞，優化廣告的關鍵字設置，確保廣告覆蓋熱門搜索。
#地理定位優化：通過地區興趣數據，調整廣告的地理定向，確保廣告只在目標市場中展示，從而提高轉化率。
#時間策略：根據不同時間段的趨勢，優化廣告投放的時間，比如在搜索高峰期投放更多廣告。

#5. 與廣告平台結合（如 Google Ads） 廣告主可以將 Pytrends 提供的搜索趨勢數據與 Google Ads、Facebook Ads 等廣告平台結合，通過 API 或手動設置，投放精准廣告。
#Google Ads：利用 Pytrends 獲取的熱門搜索詞和地理數據，廣告主可以在 Google Ads 中設置合適的關鍵字、廣告組和地區定向，優化點擊率和轉化率。
#Facebook Ads：利用 Pytrends 的興趣數據，結合 Facebook Ads 的受眾定位功能，廣告主可以針對特定的興趣群體、地理區域進行廣告定向投放。
#示例：結合 Pytrends 的廣告策略
#假設你正在為一個電子商務平台做廣告，主要推廣蘋果相關產品（iPhone、MacBook）。你可以用以下方法來優化廣告：
#獲取 iPhone 和 MacBook 的搜索趨勢：

pytrends.build_payload(kw_list=["iPhone", "MacBook"])
trend_data = pytrends.interest_over_time()
rndrCode(trend_data)
#找到最近最熱門的相關搜索詞，并將這些詞加入廣告的關鍵字列表：

related_queries = pytrends.related_queries()
for keyword, queries in related_queries.items():
    rising_queries = queries['rising']
    rndrCode(f"相關上升趨勢的搜索詞：{rising_queries}")
#根據地區定向廣告：假設 iPhone 在某些地區更受歡迎，廣告可以優先投放到這些地區。

region_interest = pytrends.interest_by_region()
top_regions = region_interest["iPhone"].nlargest(5)
rndrCode(f"iPhone 在這些地區最受歡迎：{top_regions}")
#時間段優化：在 Google Ads 中設置廣告展示時間，確保廣告在用戶搜索高峰期展示。
time_data = pytrends.interest_over_time()
peak_times = time_data[time_data["iPhone"] > time_data["iPhone"].mean()]
rndrCode(f"iPhone 的搜索高峰期：{peak_times}")
