from pytrends.request import TrendReq
#from pytrends import interest_over_time #build_payload, 
from stUtil import rndrCode
from streamlit import session_state, sidebar, text_input, radio as stRadio, multiselect
from streamlit import line_chart as lineChart
關鍵字={"按摩", "肩頸按摩", "推拿", "無痛按摩", "足底按摩", '足浴', '芳療'} # , '芳香療法'定義與按摩相關的關鍵詞

def auto_push_ad(keyWORD, trends, threshold=70):
    for keyword in keyWORD: #關鍵字
        if trends[keyword].iloc[-1] > threshold:  # 當關鍵詞趨勢超出阈值
            send_massage_ad(user_phone_number, keyword)
            rndrCode(f"已爲 {keyword} 推送廣告！")

def send_massage_ad(phone_number, keyword):
    message = client.messages.create(
        body=f"趕快預約您的 {keyword}！專業按摩服務，讓您放鬆身心。",
        from_=twilio_phone_number,
        to=phone_number
    )
    return message.sid
MENU, 表單=[], ['按摩SEO', '按摩趨勢', '電話推播', 'Twilio', '主題']
def 更新KW(kw):
  wordKW=set()
  global 關鍵字
  for k in kw: wordKW.add(k)
  關鍵字=wordKW
  #return 關鍵字

for ndx, Menu in enumerate(表單): MENU.append(f'{ndx}{Menu}')
with sidebar:
  menu=stRadio('表單', MENU, horizontal=True, index=0)
  newKW=text_input('新關鍵字', '')
  keyWORD=multiselect('關鍵字', 關鍵字, on_change=更新KW, args=(關鍵字,))
if menu==len(表單):
  pass
elif menu==MENU[3]:
  from twilio.rest import Client # Twilio API 配置信息
  account_sid = 'your_account_sid'   # 從Twilio控制台獲取
  auth_token = 'your_auth_token'     # 從Twilio控制台獲取
  twilio_phone_number = 'your_twilio_phone_number'  # Twilio購買的號碼
  client = Client(account_sid, auth_token)
  # 推送按摩廣告的函數
  # 示例：發送按摩廣告
  user_phone_number = "+886XXXXXXXXX"  # 替換爲用戶手機號碼
  send_massage_ad(user_phone_number, "肩頸按摩")
  #Step 4: 使用 Streamlit 構建廣告推送控制台
  #通過 Streamlit 構建一個簡單的控制台，用于顯示趨勢數據，并控制什麼時候向用戶推送廣告。
  st.title("台灣地區按摩廣告推播系統")
elif menu==MENU[2]:
  selected_keyword = st.selectbox("選擇推送的按摩廣告關鍵詞：", 關鍵字)
  #user_phone = st.text_input("輸入用戶的電話號碼：") # 輸入用戶電話號碼

  if st.button("推送廣告"): # 當點擊按鈕時發送廣告
      if user_phone:
          sid = send_massage_ad(user_phone, selected_keyword)
          st.success(f"廣告已發送！Twilio 消息 SID: {sid}")
      else:
          st.error("請提供有效的電話號碼")
elif menu==MENU[1]:
  rndrCode("當前按摩相關的 Google 搜索趨勢：") # 顯示趨勢數據
  按摩趨勢 = session_state['按摩趨勢']  #trndDF
  trndDF = session_state['trndDF']  #trndDF
  lineChart(trndDF) #按摩趨勢
  from time import sleep as tmSleep # 每隔一段時間檢查趨勢并自動推送廣告
  tmpDF = 按摩趨勢.interest_over_time()  # 獲取最新趨勢
  auto_push_ad(keyWORD, tmpDF, threshold=70)      # 推送廣告
  lineChart(tmpDF) #按摩趨勢
  #while True:
  #    trndDF = 按摩趨勢.interest_over_time()  # 獲取最新趨勢
  #    auto_push_ad(trndDF, threshold=70)      # 推送廣告
  #    tmSleep(3600)  # 每小時檢查一次
elif menu==MENU[0]:
  if newKW: keyWORD.append(newKW)
  趨勢 = TrendReq(hl='zh-TW', tz=-480, timeout=(10,25))
  #pytrends = TrendReq(hl='zh-TW', tz=-480, timeout=0)#hl語言 tz時區
  #keywords = ['關鍵字']#改成要查詢的關鍵字
  with sidebar:
    rndrCode(keyWORD)
  if keyWORD:
    session_state['keyWORD']=keyWORD
    趨勢.build_payload(kw_list=keyWORD, cat=0, timeframe='2023-01-01 2024-12-31', geo='TW', gprop='')   #timeframe='today 1-m', 
    session_state['按摩趨勢']=趨勢
    主題=session_state['主題']=趨勢.related_topics()
    rndrCode([主題])
    #趨勢.related_queries()

    trndDF=session_state['trndDF']=趨勢.interest_over_time() # 獲取趨勢數據
    rndrCode([主題, trndDF.head()]) # 顯示趨勢數據（可用于前端可視化）
    #pytrends.build_payload(kw_list=keywords, cat=0, timeframe='2023-01-01 2024-12-31', geo='TW', gprop='') 
    #趨勢 = TrendReq(hl='zh-TW', tz=360) # 初始化 pytrends
    #趨勢.build_payload(關鍵字, cat=0, timeframe='today 1-m', geo='TW', gprop='') # 構建查詢請求，限制爲台灣地區的趨勢
#Step 3: 設置 Twilio 進行廣告推送 使用 Twilio API 推送按摩廣告到目標用戶的手機。首先，你需要在 Twilio 注冊賬號，并獲取 Account SID 和 Auth Token。
# 選擇要推送的廣告關鍵詞
#Step 5: 自動化廣告推送 爲了實現自動化廣告推送，可以通過定時任務或者基于趨勢波動的自動推送機制來動態觸發廣告。例如，當特定關鍵詞的搜索趨勢超過某個阈值時，自動觸發廣告推送。
# 自動廣告推送邏輯
#6. 部署與集成 你可以將這個系統部署在 AWS 或其他云平台上，并通過 Streamlit 實時管理和展示廣告推送。同時結合 Twilio，可以根據 Pytrends 實時趨勢變化，自動推送個性化廣告。
#7. 擴展功能 自定義廣告內容：根據不同用戶的興趣和地區，定制不同的廣告內容。例如用戶可能對特定類型的按摩（如足底按摩或肩頸按摩）更感興趣。
#推送多種類型廣告：除了短信推送，還可以結合 Twilio 進行語音通話、WhatsApp 消息等推播。
#廣告效果追蹤：通過 Twilio 的狀態回調功能，你可以跟蹤廣告的送達情況，查看用戶是否打開或點擊了廣告。
#通過 Pytrends 提供的台灣地區的搜索趨勢數據和 Twilio 提供的推播功能，可以為用戶動態提供基于他們當前需求和搜索興趣的按摩廣告。這種廣告推送系統可以幫助你更精确地向潛在客戶發送個性化的廣告，從而提升轉化率和廣告效果。
