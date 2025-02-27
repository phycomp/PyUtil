from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from streamlit import pyplot as stPlot, plotly_chart
from sklearn.impute import SimpleImputer
from streamlit import warning, subheader
from matplotlib.pyplot import figure
from plotly.express import scatter, histogram
from sklearn.decomposition import PCA
from pandas import DataFrame

def 補值(df, method='mean'):
  imputer = SimpleImputer(strategy=method)    #'most_frequent'
  #if method == 'mean': imputer = SimpleImputer(strategy='mean')
  #elif method == 'median': imputer = SimpleImputer(strategy='median')
  #elif method == 'most_frequent':
  #  imputer = SimpleImputer(strategy='most_frequent')
  try:
    df[df.columns] = imputer.fit_transform(df[df.columns]) # 進行補值
  except:
    warning("Unknown method. Please choose mean, median or most_frequent.")
    df.fillna(df.mean(), inplace=True)  # 用均值填充缺失值
  return df

def 統計摘要(df): # 數據統計摘要
  return df.describe()

def 熱圖(df): # 相關性 相關性分析
  from seaborn import heatmap as sbrnHtmp
  fig=figure(figsize=(10, 8))
  corrMatrix = df.corr()
  sbrnHtmp(corrMatrix, annot=True, cmap='coolwarm')
  stPlot(fig)

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
