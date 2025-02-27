from pytrends.request import TrendReq
pandas import 
from pytrends import build_payload, interest_over_time
#import matplotlib.pyplot as plt

# Step 1: Initialize the pytrends object
趨勢 = TrendReq(hl='zh-TW', tz=360)

# Step 2: Define the list of keywords for SEO
keywords = ['SEO 顧問', '數位行銷', '網路行銷', '品牌行銷顧問', '關鍵字廣告']

# Step 3: Get Google Trends data for the keywords
趨勢.build_payload(keywords, cat=0, timeframe='today 12-m', geo='TW', gprop='')

# Step 4: Retrieve the interest over time data
熱門 = 趨勢.interest_over_time()

# Step 5: Plot the interest over time
plt.figure(figsize=(10,6))
for keyword in keywords:
    plt.plot(熱門.index, 熱門[keyword], label=keyword)
    
plt.title('Interest Over Time for SEO Keywords')
plt.xlabel('Date')
plt.ylabel('Interest')
plt.legend()
plt.show()

from pytrends import related_queries
qryDict = pytrends.related_queries()

for keyword in keywords:
    print(f"Related queries for {keyword}:")
    print(qryDict[keyword]['top'])
