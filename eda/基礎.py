import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from streamlit import title as stTitle, file_uploader as flUpldr, checkbox as stCheckbox, write as stWrite, plotly_chart, radio as stRadio, selectbox as stSelectbox

MENU, 表單=[], ['Profile', '補值', 'dataVis', 'dataReport', '相關性']	#, '錯綜複雜', '二十四節氣'
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
