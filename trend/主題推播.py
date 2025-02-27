from streamlit import text_input, button, subheader, title, sidebar, radio as stRadio
from stUtil import rndrCode
from streamlit import line_chart as lineChart
from pytrends.request import TrendReq

MENU, 表單=[], ['初始', '先後天', '卦爻辭', '錯綜複雜', '二十四節氣']
for ndx, Menu in enumerate(表單): MENU.append(f'{ndx}{Menu}')
with sidebar:
  menu=stRadio('表單', MENU, horizontal=True, index=0)
  srch=text_input('搜尋', '')
if menu==len(表單):
  pass
elif menu==MENU[1]:
  ad_text = text_input("Advertisement Content", "Exclusive Podcast service promotion!") # 設置廣告推送條件
  鈕 = button("Push Advertisement")
  if 鈕:
    rndrCode(["Ad Sent:", ad_text])
    # 這里可以調用 Twilio API 或其他廣告推送渠道進行推送
  #Step 2: 動態廣告推送 基於趨勢分析的結果，你可以動態推送廣告。例如，當某個關鍵詞的搜索量突然上升時，系統可以自動推送與該關鍵詞相關的廣告。以下是一個簡單的邏輯，基於搜索量的上升來觸發廣告推送：
  # 假設我們想在搜索量上升時推送廣告
  最新 = trends_data.iloc[-1]  # 獲取最新的趨勢數據
  if 最新["Podcast"] > 75:  # 如果 Podcast 的趨勢值超過 75
    # 推送廣告 (這里可以集成 Twilio 或其他消息推送服務)
    rndrCode("Trending! Sending Podcast related advertisement...")
elif menu==MENU[0]:
# 初始化 Pytrends API
  趨勢 = TrendReq(hl='zh-TW', tz=-480)
  關鍵字 = ["Podcast", "audio streaming", "Spotify", "Apple Podcast"] # 設置搜索關鍵詞
  趨勢.build_payload(關鍵字, cat=0, timeframe='today 12-m', geo='TW', gprop='') # 獲取趨勢數據
  趨勢數據 = 趨勢.interest_over_time() # 獲取趨勢數據和相關查詢
  #pytrends.related_queries()
  主題 = 趨勢.related_queries()

  title("Podcast Trends and Advertising Push") # 顯示趨勢數據的可視化
  lineChart(趨勢數據)

  # 顯示相關關鍵詞
  subheader("Related Queries for Podcast")
  rndrCode(主題['Podcast']['top'])
