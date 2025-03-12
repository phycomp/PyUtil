在streamlit 加入 carousel 商品展示功能
在 Streamlit 中加入 carousel 功能來展示商品圖片，可以利用一些支援 Carousel 的外部套件，例如 Streamlit-Components 和 Streamlit-Elements。其中 Streamlit-Elements 提供了一個更靈活的方法來嵌入自訂的 JavaScript 元件，包括 Carousel。
使用 Streamlit-Elements 實現 Carousel
    1. 安裝套件：
       pip install streamlit-elements
    2. 編寫 Carousel 程式碼： 以下是使用 Streamlit-Elements 和 Swiper.js（一種 JavaScript Carousel Library）的範例程式碼來建立 Carousel：
       import streamlit as st
       from streamlit_elements import elements, mui, html
       
       # 初始化 Streamlit 页面
       st.title("商品展示 Carousel")
       
       # 商品圖片清單
       product_images = [
           "https://example.com/image1.jpg",
           "https://example.com/image2.jpg",
           "https://example.com/image3.jpg"
       ]
       
       # 使用 Elements 组件
       with elements("carousel"):
           # 引入 Swiper CSS
           html.link(rel="stylesheet", href="https://unpkg.com/swiper/swiper-bundle.min.css")
       
           # 引入 Swiper JS
           html.script(src="https://unpkg.com/swiper/swiper-bundle.min.js")
       
           # Swiper HTML 代码
           html.div("""
           <div class="swiper-container">
               <div class="swiper-wrapper">
                   <div class="swiper-slide"><img src='{}' alt='商品 1'/></div>
                   <div class="swiper-slide"><img src='{}' alt='商品 2'/></div>
                   <div class="swiper-slide"><img src='{}' alt='商品 3'/></div>
               </div>
               <!-- 添加導航按鈕 -->
               <div class="swiper-button-next"></div>
               <div class="swiper-button-prev"></div>
           </div>
           """.format(*product_images))
       
           # 初始化 Swiper 的 JS 代码
           html.script("""
           var swiper = new Swiper('.swiper-container', {
               navigation: {
                   nextEl: '.swiper-button-next',
                   prevEl: '.swiper-button-prev',
               },
               loop: true
           });
           """)
       
程式說明
    • 圖片來源：product_images 列表包含了商品圖片的 URL，你可以將圖片替換為你想要展示的商品圖片。
    • HTML 和 JavaScript：我們利用 HTML 加入了 Swiper 的基本結構，並以 JavaScript 進行初始化，設置左右導航按鈕和循環顯示。
    • CSS 和 JavaScript 引入：我們在 HTML 中引入了 Swiper 的 CSS 和 JavaScript，讓 Carousel 能夠正常顯示。
使用 Streamlit-Components (Alt 方法)
如果你偏好其他方法，像 Streamlit-Components，可以使用它來嵌入其他 JavaScript Carousel 庫（例如 Slick 或 Tiny Slider）來進行相似操作。
更多 Streamlit-Elements 的進階範例
Streamlit-Elements 提供了靈活的方式來在 Streamlit 中嵌入各種前端元件，例如卡片、按鈕、圖表等進階組件。這裡是一些更進階的範例來展示 Streamlit-Elements 的多種應用：
1. 建立動態卡片展示
以下範例展示了如何使用 Material-UI 卡片來顯示多個商品資訊。每個卡片包含商品名稱、圖片和按鈕。
import streamlit as st
from streamlit_elements import elements, mui

# 商品資料
products = [
    {"name": "商品 A", "price": "$10", "image": "https://example.com/image1.jpg"},
    {"name": "商品 B", "price": "$20", "image": "https://example.com/image2.jpg"},
    {"name": "商品 C", "price": "$30", "image": "https://example.com/image3.jpg"},
]

st.title("動態商品卡片展示")

with elements("cards"):
    for product in products:
        with mui.Card(key=product["name"], sx={"maxWidth": 345, "margin": "20px"}):
            # 顯示圖片
            mui.CardMedia(
                component="img",
                height="140",
                image=product["image"],
                alt=product["name"]
            )
            # 顯示商品資訊
            with mui.CardContent():
                mui.Typography(product["name"], variant="h5", component="div")
                mui.Typography(product["price"], color="text.secondary")
            # 顯示操作按鈕
            with mui.CardActions():
                mui.Button("購買", size="small")
                mui.Button("加入收藏", size="small")
2. 互動式數據圖表
Streamlit-Elements 支援 Plotly 和 Highcharts 等圖表庫，可以用於繪製動態圖表。
import plotly.express as px
from streamlit_elements import elements, mui, mui_icons

# 創建範例數據
df = px.data.iris()

# 生成圖表
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")

st.title("互動式數據圖表")

with elements("chart"):
    # 放置 Plotly 圖表
    mui.Typography("Iris Data Chart", variant="h6")
    mui.plotly(fig)
3. 實現 Dashboard 面板
使用 Streamlit-Elements 可以創建完整的 Dashboard，整合卡片、圖表和按鈕組件。
import random
from streamlit_elements import elements, mui, mui_icons

st.title("Dashboard 示例")

# Dashboard 結構
with elements("dashboard"):
    # 第一行
    with mui.Grid(container=True, spacing=2):
        # 卡片 - 用戶資訊
        with mui.Grid(item=True, xs=4):
            mui.Card(
                mui.CardContent(
                    mui.Typography("用戶數", variant="h5"),
                    mui.Typography(f"{random.randint(1000, 5000)} 人", color="text.secondary")
                )
            )
        
        # 卡片 - 每日訪問量
        with mui.Grid(item=True, xs=4):
            mui.Card(
                mui.CardContent(
                    mui.Typography("每日訪問量", variant="h5"),
                    mui.Typography(f"{random.randint(500, 1500)} 次", color="text.secondary")
                )
            )

        # 卡片 - 銷售額
        with mui.Grid(item=True, xs=4):
            mui.Card(
                mui.CardContent(
                    mui.Typography("今日銷售額", variant="h5"),
                    mui.Typography(f"${random.randint(1000, 10000)}", color="text.secondary")
                )
            )

    # 第二行 - 圖表
    with mui.Grid(container=True, spacing=2, sx={"marginTop": "20px"}):
        with mui.Grid(item=True, xs=12):
            st.line_chart([random.random() for _ in range(100)], width=0, height=300)
4. 文件預覽嵌入（使用 PDF.js）
您可以使用 Streamlit-Elements 與 PDF.js 嵌入 PDF 檔案預覽功能。
from streamlit_elements import elements, html

st.title("PDF 文件預覽")

with elements("pdf_preview"):
    # 引入 PDF.js
    html.link(rel="stylesheet", href="https://mozilla.github.io/pdf.js/web/viewer.css")
    html.script(src="https://mozilla.github.io/pdf.js/build/pdf.js")

    # 顯示 PDF 文件
    html.div("""
    <iframe
        src="https://mozilla.github.io/pdf.js/web/viewer.html?file=https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        width="100%"
        height="600px"
    >
    </iframe>
    """)
這些進階範例展示了 Streamlit-Elements 的多樣性，允許您通過整合 JavaScript 元件、資料視覺化和互動元件來建構功能豐富的應用程式。Streamlit-Elements 能幫助您在 Streamlit 中實現更多自訂的前端體驗，非常適合 Dashboard 和展示用途。
在 streamlit 加入留言板功能 評論 地圖 增加互動性
在 Streamlit 中添加留言板功能、評論區以及地圖，可以增強互動性，並提升用戶體驗。以下是如何實現這些功能的簡單指南。
1. 加入留言板功能
留言板功能可以透過 Streamlit 的 text_input 和 session_state 來保存使用者的留言。您也可以選擇將留言存入外部資料庫，如 Google Sheets 或 SQLite。
import streamlit as st

# 建立留言板標題
st.title("留言板")

# 使用 session_state 存放留言
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 新增留言
new_message = st.text_input("留下您的留言")
if st.button("提交留言"):
    if new_message:
        st.session_state["messages"].append(new_message)
        st.success("留言已提交！")
        st.experimental_rerun()

# 顯示留言
st.subheader("所有留言")
for i, msg in enumerate(st.session_state["messages"]):
    st.write(f"{i + 1}. {msg}")
2. 添加評論區（帶有評分）
評論區可以透過 st.radio 或 st.slider 來收集評分，並且將評分和留言一起顯示。
import datetime

st.title("評論區")

# 建立評論區的資料存放
if "reviews" not in st.session_state:
    st.session_state["reviews"] = []

# 獲取使用者的評分和留言
rating = st.slider("請給予評分", 1, 5, step=1)
review_message = st.text_area("留下您的評論")

if st.button("提交評論"):
    if review_message:
        review = {
            "rating": rating,
            "message": review_message,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state["reviews"].append(review)
        st.success("評論已提交！")
        st.experimental_rerun()

# 顯示所有評論
st.subheader("用戶評論")
for review in st.session_state["reviews"]:
    st.write(f"評分: {review['rating']} 星")
    st.write(f"留言: {review['message']}")
    st.write(f"時間: {review['time']}")
    st.write("---")
3. 增加地圖功能
地圖功能可以通過 st.map 或 pydeck_chart 來實現。用戶可以在地圖上互動，或標記自己所在的地點。這裡展示了一個簡單的地圖，顯示指定位置。
import pandas as pd

# 標記位置資料
st.title("互動地圖")

locations = pd.DataFrame({
    'latitude': [25.0340, 24.8060, 25.0418],
    'longitude': [121.5645, 121.5379, 121.525],
    'location_name': ['台北101', '故宮博物院', '士林夜市']
})

# 顯示地圖
st.map(locations)

# 可選擇顯示詳細位置
selected_location = st.selectbox("選擇地點以查看詳細資訊", locations['location_name'])
location_data = locations[locations['location_name'] == selected_location]
st.write(f"{selected_location} 的位置：")
st.write(location_data)
4. 結合留言、評論和地圖
如果要在地圖上顯示地點評論或留言，可以將上述各個功能結合在一起。例如，可以在選擇某個地點後顯示該地點的留言區，並讓用戶對該地點進行評論。
# 結合地圖選擇與留言評論功能
st.title("地點評論互動")

selected_location = st.selectbox("選擇地點進行評論", locations['location_name'])
st.write(f"{selected_location} 的位置：")
location_data = locations[locations['location_name'] == selected_location]
st.map(location_data)

# 顯示特定地點的留言和評論
st.subheader(f"{selected_location} 的留言與評論")
if f"{selected_location}_reviews" not in st.session_state:
    st.session_state[f"{selected_location}_reviews"] = []

# 獲取留言和評分
rating = st.slider("請給予評分", 1, 5, step=1)
review_message = st.text_area("留下您的評論")

if st.button("提交評論"):
    if review_message:
        review = {
            "rating": rating,
            "message": review_message,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state[f"{selected_location}_reviews"].append(review)
        st.success("評論已提交！")
        st.experimental_rerun()

# 顯示該地點的所有評論
for review in st.session_state[f"{selected_location}_reviews"]:
    st.write(f"評分: {review['rating']} 星")
    st.write(f"留言: {review['message']}")
    st.write(f"時間: {review['time']}")
    st.write("---")
這樣的佈局允許使用者選擇特定地點後，直接在地圖下方查看或提交該地點的評論。這些功能增加了互動性，讓應用程式更符合用戶需求。
