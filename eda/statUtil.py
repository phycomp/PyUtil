from pandas import read_csv
from pandas_profiling import ProfileReport
from streamlit import session_state, radio as stRadio, file_uploader as flUpldr
from stUtil import rndrCode

MENU, 表單=[], ['dset', 'Skew/Kurt', 'ProfileReport', 'corr分析', 'PCA分析', '統計分析'] #miscPlot
for ndx, Menu in enumerate(表單): MENU.append(f'{ndx}{Menu}')
with sidebar:
  menu=stRadio('表單', MENU, horizontal=True, index=0)
  #srch=text_input('搜尋', '')
if menu==len(表單):
  pass
elif menu==MENU[5]:
  from scipy import stats
  data=session_state['data']
  z_scores = stats.zscore(data['age']) # Calculate the Z-scores

  threshold = 3 # Define a threshold for outlier detection
  outliers = data[abs(z_scores) > threshold] # Identify outliers
  rndrCode(outliers) # Display the outliers
elif menu==MENU[4]:
  from sklearn.decomposition import PCA # Let's assume we have preprocessed our data and it's stored in 'X'
  pca = PCA(n_components=2)     # Perform PCA
  pca_result = pca.fit_transform(X)
  plt.scatter(pca_result[:, 0], pca_result[:, 1], color='purple')   # Plot the PCA results
  plt.xlabel('Principal Component 1')
  plt.ylabel('Principal Component 2')
  plt.title('PCA Results')
  plt.show()
elif menu==MENU[3]:
  dsetDF=session_state['data']
  相依性 = dsetDF.corr()
  plt.figure(figsize=(8, 6)) # Visualize the correlation matrix as a heatmap
  sns.heatmap(相依性, annot=True, cmap='coolwarm')
  plt.title('Correlation Matrix')
  plt.show()
elif menu==MENU[2]: #ProfileReport
  from streamlit_pandas_profiling import st_profile_report as profileReport
  from pandas_profiling import ProfileReport
  df = session_state['data']#read_csv("your_dataset.csv") # 加載資料
  profile = ProfileReport(df, explorative=True) # 生成報告
  profileReport(profile) # 在 Streamlit 中顯示
elif menu==MENU[1]:     #'Skew/Kurt'
  dset=session_state['data']
  skewness = dset.skew()
  kurtosis = dset.kurt()
  for column in dset.select_dtypes(include=['float', 'int']).columns: # 視覺化數據分佈
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f"{column} Distribution")
    plt.show()
elif menu==MENU[0]:
  upldedFLE = flUpldr("Upload the dset file", type=["csv"])
  if upldedFLE:
    #data = read_csv(uploaded_file)
    dsetDF=session_state['data']=read_csv(upldedFLE) # 加載資料
    profile = ProfileReport(dsetDF, explorative=True) # 生成分析報告
    profile.to_file("output_report.html")
    rndrCode(["### Dataset Preview", dsetDF.head()])
# 將報告輸出為 HTML 文件 在 Streamlit 中顯示報告 如果你想將 Pandas Profiling 報告嵌入到 Streamlit 應用中，可以使用 streamlit-pandas-profiling。 pip install streamlit-pandas-profiling
#Pandas Profiling 的優勢
#節省時間：它自動生成詳細的 EDA 報告，無需手動執行每一步分析，讓你快速掌握數據特性。
#互動式報告：生成的 HTML 報告是互動式的，可以方便地檢查具體數據和視覺化結果。
#全面性：不僅包括數據的基本統計量，還涵蓋了缺失值、異常值、相關性等多方面的信息。
#易於集成：可以與 Streamlit 等應用程序無縫集成，讓數據分析過程更加靈活。
#總結 Pandas Profiling 通過自動化資料分析，大大提高了探索性資料分析的效率和準確性。它不僅能生成詳細的報告，還幫助用戶快速識別資料中的問題（如缺失值、異常值等），從而為進一步的數據處理和模型構建提供有效支持。
"""
分析原理 pandas profiling

Pandas Profiling 是一個 Python 庫，用於自動生成數據集的全面分析報告，這在探索性資料分析 (EDA) 中非常有用。它會生成一個互動式的 HTML 報告，提供有關數據分佈、數據品質、變數關聯性等詳細信息。分析報告涵蓋了數據的多個層面，包括數值統計、缺失值、異常值、相關性、偏度等。

Pandas Profiling 的分析原理
數據的基本信息 報告首先總覽數據集的基本統計信息，如資料形狀（行數和列數）、資料類型、缺失值、重複值等。 此步驟可以快速幫助用戶瞭解數據的大小、結構和整體健康情況。

總行數
總列數
數值列、類別列、時間列的數量
重複行的比例
每個變數的摘要

對每個變數進行詳細的描述統計，涵蓋如下信息：
計數值：非空值的數量
唯一值：每個變數中唯一值的數量
最頻繁的值：出現次數最多的值（類別型變數的眾數）
缺失值百分比：缺失數據的比例
平均值、中位數、標準差、極值：數值型變數的基本統計量
例子：

- Column A:
  - Non-null count: 950
  - Unique values: 5
  - Most frequent value: 2 (occurs 230 times)
  - Mean: 5.6, Median: 5.5, Std: 2.1
數據分佈

數值型變數：Pandas Profiling 會自動生成直方圖、箱形圖等，顯示數據的分佈、偏度（skewness）和峰度（kurtosis）。
類別型變數：對類別型變數，會顯示條形圖，展示每個類別的頻次。
時間序列：如果數據有日期時間類型，報告會包括時間分佈的圖表。

偏度（Skewness）：顯示資料分佈的對稱性。
峰度（Kurtosis）：顯示資料的尖峰程度，幫助識別分佈的尾部是否異常。
相關性分析

Pandas Profiling 通過計算數值變數之間的 皮爾森相關係數 或 斯皮爾曼相關係數 來衡量變數之間的線性關聯性。還會用熱圖（Heatmap）來可視化相關性矩陣，幫助用戶快速找到高度相關的變數。
- Correlation between A and B: 0.75 (strong positive correlation)
- Correlation between A and C: -0.45 (moderate negative correlation)
異常值檢測 報告會顯示數據中的異常值（outliers），特別是使用箱形圖法（Box plot）來檢測。這些異常值可能是數據錯誤或者極端值，對後續分析有重大影響。
列 B 中的異常值在 Box plot 中以點的形式標出。
缺失值分析

報告會檢查每個變數中的缺失值數量和比例，並顯示一個缺失值的熱圖（Missing Values Heatmap），幫助用戶快速識別資料品質問題。如果缺失值非常多，會給出相應的處理建議。
列 D 有 30% 的缺失值，報告中會顯示該變數的缺失值分佈。
高級特徵 Pandas Profiling 還會自動檢測資料中的高級特徵，如常量列（每個值都一樣的列）、高基數（很多唯一值的類別型變數）以及重複列（完全相同的兩列）。這些特徵可能需要在後續分析中移除或進行特徵工程。
使用 Pandas Profiling 的流程
安裝 Pandas Profiling
可以通過 pip 安裝 Pandas Profiling：

pip install pandas-profiling

"""
