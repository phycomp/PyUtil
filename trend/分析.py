from pytrends.request import TrendReq

pytrends = TrendReq(hl='en-US', tz=360) # 初始化 Pytrends
#(2) 獲取基於用戶興趣的熱門趨勢 根據用戶的興趣愛好（例如電子產品、服裝等），可以使用 Pytrends 來分析特定類別的熱門趨勢。

# 關鍵字列表：根據用戶興趣進行個性化的搜索
keywords = ["iPhone", "MacBook", "smartwatch"]

# 獲取趨勢數據
pytrends.build_payload(kw_list=keywords)

# 獲取興趣隨時間的變化
interest_over_time_df = pytrends.interest_over_time()
print(interest_over_time_df.head())
#(3) 獲取地理定向數據 如果你想根據用戶的位置進行個性化廣告推送，可以通過 Pytrends 獲取特定地區的熱門趨勢。

# 獲取興趣按地區分布
interest_by_region_df = pytrends.interest_by_region()
print(interest_by_region_df.head())

# 獲取最受歡迎的地區
top_regions = interest_by_region_df.nlargest(5, 'iPhone')
print(f"iPhone 熱門地區：\n{top_regions}")
#(4) 獲取相關搜索詞 可以通過相關搜索詞擴展廣告推送的關鍵詞覆蓋面，使得推送的廣告與用戶搜索的主題更加相關，從而提高廣告的個性化程度。

# 獲取相關的上升趨勢查詢
related_queries = pytrends.related_queries()
for keyword, queries in related_queries.items():
    rising_queries = queries['rising']
    print(f"與 {keyword} 相關的上升趨勢搜索詞：{rising_queries}")
#2. 整合推播廣告平台 將 Pytrends 提供的個性化數據整合到推播廣告平台中，例如 Firebase Cloud Messaging (FCM) 或 OneSignal，可以根據實時趨勢推送個性化廣告。

#(1) Firebase Cloud Messaging (FCM) 推送廣告 使用 Pytrends 分析出的數據，你可以將其結合到 Firebase 推播通知的廣告內容中。例如：

import firebase_admin
from firebase_admin import messaging

# 初始化 Firebase 應用
firebase_admin.initialize_app()

# 生成推送通知消息
def send_push_notification(title, body, token):
    message = messaging.Message(
        notification=messaging.Notification(
            title=title,
            body=body,
        ),
        token=token,
    )

    # 發送消息
    response = messaging.send(message)
    print('Successfully sent message:', response)

# 獲取當前熱門的 iPhone 相關趨勢，并推送廣告
trending_search = "iPhone 15 Pro Max"
send_push_notification(f"新款 {trending_search} 上市！", "立即購買，享受優惠！", "用戶設備的token")
#(2) OneSignal 推送廣告 OneSignal 是另一款推播廣告平台，可以使用 Pytrends 分析出的個性化關鍵詞生成通知內容，來提高廣告的相關性和吸引力。

import requests

# 設置 OneSignal API
ONESIGNAL_APP_ID = "your-onesignal-app-id"
ONESIGNAL_REST_API_KEY = "your-onesignal-rest-api-key"

def send_onesignal_notification(title, content, player_id):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Basic {ONESIGNAL_REST_API_KEY}",
    }

    payload = {
        "app_id": ONESIGNAL_APP_ID,
        "include_player_ids": [player_id],
        "headings": {"en": title},
        "contents": {"en": content},
    }

    response = requests.post("https://onesignal.com/api/v1/notifications", json=payload, headers=headers)
    print(response.status_code, response.json())

# 通過 Pytrends 獲取的熱門關鍵詞推送廣告
send_onesignal_notification("新品 iPhone 搶購！", "iPhone 15 系列現已發布，快來選購！", "用戶ID")
#3. 實現動態推播廣告策略
#(1) 實時監測趨勢變化 通過定期監測趨勢變化，推播廣告可以做到動態調整。例如，在節假日、購物季等高峰期，推送與當時熱門趨勢相關的廣告。

import time

# 定期檢查趨勢，并推送相應廣告
while True:
    # 檢查是否有新的上升趨勢
    rising_trends = pytrends.related_queries()["iPhone"]["rising"]
    if rising_trends is not None:
        # 推送與新趨勢相關的廣告
        new_trend = rising_trends.iloc[0]["query"]
        send_push_notification(f"搶購熱潮：{new_trend}！", "新款手機火熱促銷中，立即購買！", "用戶設備的token")

    # 每 24 小時更新一次
    time.sleep(86400)
#(2) 定向用戶群體 根據用戶的地理位置或興趣定向推播廣告。例如，用戶位於特定的城市或地區，可以推送與該地區趨勢相關的廣告。

# 只針對特定地區的用戶進行推送
top_regions = interest_by_region_df.nlargest(5, 'iPhone')
if user_location in top_regions.index:
    send_push_notification("本地區的特價促銷！", "立享優惠，限時折扣！", "用戶設備的token")
