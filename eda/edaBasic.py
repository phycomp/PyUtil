import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from streamlit import title as stTitle, file_uploader as flUpldr, checkbox as stCheckbox, write as stWrite, plotly_chart, radio as stRadio, selectbox as stSelectbox

MENU, 表單=[], ['Profile', '補值', 'Skew', 'dataVis', 'dataReport', '相關性']	#, '錯綜複雜', '二十四節氣'
for ndx, Menu in enumerate(表單): MENU.append(f'{ndx}{Menu}')
with sidebar:
  menu=stRadio('表單', MENU, horizontal=True, index=0)
  #srch=text_input('搜尋', '')
if menu==len(表單):
  if stCheckbox("Show Histogram"):
    fig, ax = plt.subplots()
    sns.histplot(data[selected_x], kde=True, ax=ax)
    st.pyplot(fig)
  if stCheckbox("Show Scatter Plot"):
    fig = px.scatter(data, x=selected_x, y=selected_y)
    st.plotly_chart(fig)

  if stCheckbox("Show Box Plot"):
    fig, ax = plt.subplots()
    sns.boxplot(x=data[selected_x], ax=ax)
    st.pyplot(fig)
  #相關性矩陣： 使用 Seaborn 的 heatmap 來繪製相關性矩陣，幫助用戶發現數據中變數之間的線性關聯性。

  if stCheckbox("Show Correlation Matrix"):
    fig, ax = plt.subplots()
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
  #自動生成 EDA 報告： 通過 Pandas Profiling 自動生成 EDA 報告，包含詳細的資料分佈、相關性、遺漏值分析等信息。
  if stCheckbox("Generate EDA Report"):
    profile = ProfileReport(data, explorative=True)
    st_profile_report(profile)
  elif menu==MENU[1]:
    if stCheckbox("Show Histogram"): # Histogram
      fig, ax = plt.subplots()
      sns.histplot(data[selected_x], kde=True, ax=ax)
      st.pyplot(fig)

    if stCheckbox("Show Scatter Plot"): # Scatter Plot
      fig = px.scatter(data, x=selected_x, y=selected_y)
      st.plotly_chart(fig)

    if stCheckbox("Show Box Plot"): # Box Plot
      fig, ax = plt.subplots()
      sns.boxplot(x=data[selected_x], ax=ax)
      st.pyplot(fig)

    if stCheckbox("Show Correlation Matrix"): # Correlation Matrix
      fig, ax = plt.subplots()
      corr_matrix = data.corr()
      sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
      st.pyplot(fig)
    if stCheckbox("Generate EDA Report"): # Auto EDA Report with Pandas Profiling
      profile = ProfileReport(data, explorative=True)
      st_profile_report(profile)
elif menu==MENU[0]:
  stTitle("Advanced EDA Framework with Streamlit") # Streamlit App Title
  uploaded_file = flUpldr("Upload your CSV file", type=["csv"])
  upldedFLE = flUpldr("Upload your CSV file", type=["csv"])
  if upldedFLE is not None:
      data = pd.read_csv(uploaded_file)
      st.write("### Dataset Preview", data.head())
      if stCheckbox("Show Dataset Information"): # Basic Data Information
          stWrite("Data Shape: ", data.shape)
          stWrite("Data Types:", data.dtypes)
          stWrite("Missing Values:", data.isnull().sum())

      if stCheckbox("Show Statistical Summary"): # Statistical Summary
          stWrite(data.describe())

      stWrite("### Data Visualization")    #Select Columns for Visualization
      columns = data.columns.tolist()
      selected_x = stSelectbox("Select X-Axis Variable", columns)
      selected_y = stSelectbox("Select Y-Axis Variable", columns)
#基本統計分析： 提供基本的統計摘要，使用 Pandas 的 describe() 函數，顯示資料的平均值、標準差、最小值等統計資訊。

  #Profile
"""
使用 streamlit 開發advanced EDA 分析資料 framework
使用 Streamlit 開發一個進行 advanced EDA（高級探索性資料分析）的框架，可以快速構建一個具有互動功能的 Web 應用，來展示和分析資料。Streamlit 提供了非常簡單的 API，可以與 Pandas、Matplotlib、Seaborn 等工具無縫集成。

以下是構建一個 Advanced EDA Framework 的步驟，包括資料加載、可視化、統計分析、相關性分析和自動化報告生成等功能。

1. 準備工作 首先，安裝所需的包：

pip install streamlit pandas matplotlib seaborn plotly pandas-profiling
2. 開發流程 我們將構建一個 Streamlit 應用，實現以下功能：

資料加載與顯示
基本統計分析
資料可視化（直方圖、散點圖、箱形圖等）
相關性矩陣與熱圖
自動化 EDA 報告生成

"""


#3. 各功能說明 資料加載與預覽： 使用 st.file_uploader 函數加載 CSV 文件，並顯示數據的前幾行，方便用戶了解資料結構。

"""
資料可視化：
直方圖：通過 seaborn.histplot 生成資料的分佈圖，檢查資料的分佈是否符合正態。
散點圖：使用 Plotly 繪製交互式的散點圖，查看兩個變數之間的關聯性。
箱形圖：用 Seaborn 的 boxplot 檢查數據的異常值和分佈範圍。

"""
"""
4. 如何運行應用
在你的終端中運行以下命令，啟動 Streamlit 應用：
streamlit run app.py
應用將打開一個瀏覽器窗口，讓你能夠上傳資料並進行探索性數據分析。這個框架允許你在網頁上互動式地分析資料，快速發現數據中的模式、異常值和相關性。
5. 可擴展性
這個框架可以根據需求進行擴展，例如：
添加更多自定義的圖表和分析方法。
增加數據處理步驟，如資料清洗、特徵工程等。
集成更多的高級統計分析和機器學習功能。
"""
