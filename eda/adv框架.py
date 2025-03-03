from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from streamlit import file_uploader as flUpldr, pyplot as stPlot, plotly_chart, subheader, warning
from plotly.express import scatter, histogram

def 補值(df, method='mean'):
  from sklearn.impute import SimpleImputer
  if method == 'mean': imputer = SimpleImputer(strategy='mean')
  elif method == 'median': imputer = SimpleImputer(strategy='median')
  elif method == 'most_frequent':
    imputer = SimpleImputer(strategy='most_frequent')
    df[df.columns] = imputer.fit_transform(df[df.columns]) # 進行補值
  else:
    warning("Unknown method. Please choose 'mean', 'median' or 'most_frequent'.")
    df.fillna(df.mean(), inplace=True)  # 用均值填充缺失值
  return df

def 統計摘要(df): # 數據統計摘要
  return df.describe()

def 相關性(df): # 相關性分析
  from seaborn import heatmap as sbrnHtmp
  plt.figure(figsize=(10, 8))
  corrMatrix = df.corr()
  sbrnHtmp(corrMatrix, annot=True, cmap='coolwarm')
  stPlot(plt)

def 繪分佈圖(df): # 數據分佈
  subheader("Distribution Plots")
  for col in df.select_dtypes(include='number').columns:
    fig = histogram(df, x=col, title=f'Distribution of {col}')
    plotly_chart(fig)

def anaPCA(df): # PCA 分析
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(df.select_dtypes(include='number'))
  pca = PCA(n_components=2)
  principal_components = pca.fit_transform(scaled_data)
  return DataFrame(data=principal_components, columns=['PC1', 'PC2'])

def KMeans聚類(df, n_clusters): # kmeans_clustering KMeans 聚類
    kmeans = KMeans(n_clusters=n_clusters)
    return kmeans.fit_predict(df)

# DBSCAN 聚類
def DBSCAN聚類(df, eps, min_samples):  #dbscan_clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(df)

# 主應用程序
st.title("Advanced EDA Dashboard")

uploaded_file = flUpldr("Upload your CSV file", type=["csv"])
補值=['mean', 'median', 'most_frequent']
if uploaded_file: # 數據上傳
    df = pd.read_csv(uploaded_file)
    rndrCode("### Data Preview")
    st.dataframe(df.head())
    imputation_method = st.selectbox("Select imputation method", options=補值) # 選擇補值方法
    # 數據預處理
    if st.checkbox("Preprocess Data"):
        df = 預處理(df, method=imputation_method)   #preprocess_data
        rndrCode("Data after preprocessing:")
        st.dataframe(df.head())
    
    # 數據摘要
    if st.checkbox("Show Data Summary"):
        summary = data_summary(df)
        rndrCode(summary)

    # 相關性熱圖
    if st.checkbox("Show Correlation Heatmap"):
        correlation_heatmap(df)

    # 數據分佈
    if st.checkbox("Show Distribution Plots"):
        plot_distribution(df)

    # PCA
    if st.checkbox("Perform PCA"):
        pca_result = anaPCA(df)
        rndrCode("PCA Result:")
        st.dataframe(pca_result)

        # PCA 可視化
        fig = scatter(pca_result, x='PC1', y='PC2', title='PCA Result')
        st.plotly_chart(fig)

    # KMeans 聚類
    if st.checkbox("KMeans Clustering"):
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)
        kmeans_labels = KMeans聚類(df.select_dtypes(include='number'), n_clusters)    #kmeans_clustering
        df['KMeans Cluster'] = kmeans_labels
        rndrCode("KMeans Clustering Results:")
        st.dataframe(df.head())

        fig = scatter(df, x=df.columns[0], y=df.columns[1], color='KMeans Cluster', title='KMeans Clustering Results')
        st.plotly_chart(fig) # KMeans 可視化

    if st.checkbox("DBSCAN Clustering"):
        eps = st.slider("Select epsilon (eps)", 0.1, 10.0, 0.5)
        min_samples = st.slider("Select minimum samples", 1, 10, 5)
        dbscan_labels = DBSCAN聚類(df.select_dtypes(include='number'), eps, min_samples) # DBSCAN 聚類dbscan_clustering
        df['DBSCAN Cluster'] = dbscan_labels
        rndrCode("DBSCAN Clustering Results:")
        df.head()
        #dataframe()

        # DBSCAN 可視化
        fig = scatter(df, x=df.columns[0], y=df.columns[1], color='DBSCAN Cluster', title='DBSCAN Clustering Results')
        plotly_chart(fig)
