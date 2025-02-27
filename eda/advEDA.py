from streamlit import file_uploader as flUpldr, subheader, warning, title, checkbox, radio as stRadio, text_input, sidebar, session_state, plotly_chart, columns as stCLMN, slider
from stUtil import rndrCode
from pandas import read_csv, DataFrame
from edaUtil import 補值, 統計摘要, 熱圖, 繪分佈圖, anaPCA, KMeans聚類, DBSCAN聚類
from matplotlib.pyplot import figure
from plotly.express import scatter, histogram

#MENU, 表單=[], ['預處理', '先後天', 'PCA', '錯綜複雜', '二十四節氣']
MENU, 流程=[], ['skew/kurt', '摘要', '熱圖', '分佈', 'PCA', 'KMeans', 'DBSCAN', 'profileReport', '遞廻分類']  #預處理
資料集=["load_iris", "load_breast_cancer", "load_diabetes", "load_digits", "load_files", "load_linnerud", "load_sample_image", "load_sample_images", "load_svmlight_file", "load_svmlight_files", "load_wine", 'load_boston'] #boston = load_boston()

補值方式=['mean', 'median', 'most_frequent']

for ndx, Menu in enumerate(流程): MENU.append(f'{ndx}{Menu}')
with sidebar:
  #menu=stRadio('表單', MENU, horizontal=True, index=0)
  menu=stRadio('流程', MENU, horizontal=True, index=0)
  補資料=stRadio("補值", 補值方式, horizontal=True, index=0) # 選擇補值方法   Select imputation method
  子集=stRadio("資料集", 資料集, horizontal=True, index=0) # 選擇補值方法   Select imputation method
  #srch=text_input('搜尋', '')
  upldFILE = flUpldr("上傳檔案", type=["csv"])
leftPane, rightPane=stCLMN([3,7])
if upldFILE:    #數據上傳
  dsetDF=session_state['dsetDF']=read_csv(upldFILE)
  #rndrCode("### Data Preview")
  dsetDF.head()
  #dataframe()
if 子集:
  載入資料=f'from sklearn.datasets import {子集}'
  exec(載入資料)
  CMD=f"dset=session_state['dset']={子集}()"
  #rndrCode(CMD)
  exec(CMD)  #f"dsetDF=session_state['dsetDF']={dset}()"
  #from sklearn.datasets import load_iris
  #iris = session_state['iris']=load_iris() # Load Iris dataset
  dsetDF=session_state['dsetDF']=DataFrame(dset['data'], columns=dset['feature_names'])
  dsetDF['target'] = dset['target']
  from numpy import array
  #dset=eval(str(dset))
  #from json import loads as jsnLoads
  #dsetDF=jsnLoads(str(dsetDF))
  dsetDF = 補值(dsetDF, method=補資料)   #preprocess_data 數據預處理 imputation_method
  with leftPane:
    dsetDF
  ## Pairplot for iris dataset Look at the first few rows
  #irisDF.head()
  #irisDF
  #pairPlot=sbrnPairplot(irisDF, hue='target', markers=["o", "s", "D"])
  #rndrCode(pairPlot)
  #stPlot(pairPlot)  #irisDF, plotly_chart
#plt.show()
  #rndrCode(['menu', menu])
  if menu==len(MENU):#checkbox("Show Data Summary"):
    pass
  elif menu==MENU[8]:#"DBSCAN Clustering"
    from rcrsv分類 import rcrsvKMeans, rcrsvDBSCAN, vis分群
    dsetDF=session_state['dsetDF']
    #rcrsvKMeans(dsetDF)
    次群 = rcrsvKMeans(dsetDF) #通用分群recursive_kmeans
    次群II = rcrsvDBSCAN(dsetDF) #通用分群recursive_kmeans
    rndrCode([次群, 次群II])
    vis分群(dsetDF, 次群II) # 可視化某一層的遞歸分群結果
    #recursive_dbscan_clusters = rcrsvDBSCAN(numeric_data)
  elif menu==MENU[6]:#"DBSCAN Clustering"
    with leftPane:
      eps = slider("Select epsilon (eps)", 0.1, 10.0, 0.5)
      min_samples = slider("Select minimum samples", 1, 10, 5)
      dbscan_labels = DBSCAN聚類(dsetDF.select_dtypes(include='number'), eps, min_samples) # DBSCAN 聚類dbscan_clustering
      dsetDF['DBSCAN Cluster'] = dbscan_labels
      rndrCode(["DBSCAN Clustering Results:", dsetDF.head()])
    #dataframe()
    with rightPane:
      # DBSCAN 可視化
      fig = scatter(dsetDF, x=dsetDF.columns[0], y=dsetDF.columns[1], color='DBSCAN Cluster', title='DBSCAN Clustering Results')
      plotly_chart(fig)
  elif menu==MENU[5]:#"KMeans Clustering"
    with leftPane:
      n_clusters = slider("Select number of clusters", 2, 10, 3)
      標籤 = KMeans聚類(dsetDF.select_dtypes(include='number'), n_clusters)    #kmeans_clustering KMeans 聚類
      dsetDF['KMeans Cluster'] = 標籤
      rndrCode(["KMeans Clustering Results:", dsetDF.head()])
    #dataframe()
    with rightPane:
      fig = scatter(dsetDF, x=dsetDF.columns[0], y=dsetDF.columns[1], color='KMeans Cluster', title='KMeans Clustering Results')
      plotly_chart(fig) # KMeans 可視化
  elif menu==MENU[4]:#PCA分析
    #dsetDF=session_state['dsetDF']
    #dset=session_state['dset']
    with leftPane:
      pcaDF = anaPCA(dsetDF) # PCA
      rndrCode(["PCA Result:", pcaDF])
      #dataframe()
    with rightPane:
      fig = scatter(pcaDF, x='PC1', y='PC2', title='PCA Result')
      plotly_chart(fig) # PCA 可視化 使用 PCA 進行降維并繪制二維散點圖 PCA（主成分分析）可以幫助將數據從高維空間映射到低維空間，方便可視化 Perform PCA for dimensionality reduction
  #elif menu==MENU[1]:
    #pca = PCA(n_components=2)
    #irisPCA = pca.fit_transform(dset['data'])
    # Create a DataFrame with the PCA results and target labels
    #pcaDF = DataFrame(irisPCA, columns=['PCA1', 'PCA2'])
    #pcaDF['target']=irisDF['target']

    #分散圖=figure(figsize=(8, 6)) # Plot the PCA results
    ##fig = plt.figure(figsize=(10, 4))
    #scatterplot(x='PCA1', y='PCA2', hue='target', palette='deep', data=pcaDF)
    #stPlot(分散圖)
    #plt.title('PCA of Iris Dataset')
    #plt.show()
  elif menu==MENU[3]:# "Show Distribution Plots"
    with rightPane:
      繪分佈圖(dsetDF) # 數據分佈
  elif menu==MENU[2]:# "Show Correlation Heatmap"
    with rightPane:
      熱圖(dsetDF) # 相關性熱圖
  elif menu==MENU[1]:#checkbox("Show Data Summary"):
    with rightPane:
      summary=統計摘要(dsetDF) # 數據摘要
      rndrCode(summary)
  elif menu==MENU[0]:#checkbox("Show Data Summary"):
    'skewness/kurtosis'
