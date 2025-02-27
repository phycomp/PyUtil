from seaborn import pairplot as sbrnPairplot, scatterplot
from matplotlib.pyplot import figure
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from pandas import DataFrame
from streamlit import plotly_chart, pyplot as stPlot, sidebar, radio as stRadio, session_state
from stUtil import rndrCode

MENU, 表單=[], ['pairPlot', 'PCA', '卦爻辭', '錯綜複雜', '二十四節氣']
for ndx, Menu in enumerate(表單): MENU.append(f'{ndx}{Menu}')
#2. 創建基本分布圖表 (Profiling) 我們可以使用 Seaborn 來創建分類數據的基本分布圖表：

with sidebar:
  menu=stRadio('表單', MENU, horizontal=True, index=0)
if menu==len(MENU):
  pass
elif menu==MENU[0]:
  iris = session_state['iris']=load_iris() # Load Iris dataset
  irisDF = session_state['irisDF']=DataFrame(iris['data'], columns=iris['feature_names'])
  irisDF['target'] = iris['target']

  # Pairplot for iris dataset Look at the first few rows
  irisDF.head()
  irisDF
  pairPlot=sbrnPairplot(irisDF, hue='target', markers=["o", "s", "D"])
  rndrCode(pairPlot)
  stPlot(pairPlot)  #irisDF, plotly_chart
#plt.show()

elif menu==MENU[1]: #3. 使用 PCA 進行降維并繪制二維散點圖 PCA（主成分分析）可以幫助將數據從高維空間映射到低維空間，方便可視化 Perform PCA for dimensionality reduction
  irisDF = session_state['irisDF']
  irisData=session_state['iris']
  rndrCode([irisDF.head()])
  pca = PCA(n_components=2)
  irisPCA = pca.fit_transform(irisData['data'])
  # Create a DataFrame with the PCA results and target labels
  pcaDF = DataFrame(irisPCA, columns=['PCA1', 'PCA2'])
  pcaDF['target']=irisDF['target']

  分散圖=figure(figsize=(8, 6)) # Plot the PCA results
  #fig = plt.figure(figsize=(10, 4))
  scatterplot(x='PCA1', y='PCA2', hue='target', palette='deep', data=pcaDF)
  stPlot(分散圖)
  #plt.title('PCA of Iris Dataset')
  #plt.show()
