"""
使用 recursive 進行通用分群
使用 遞歸 (recursive) 方法進行通用分群，通常涉及遞歸調用分群算法，將數據分爲多個層級的簇。這個方法適用于一些層次聚類（Hierarchical Clustering）或分割型的分群方法，例如 Recursive KMeans 或 Recursive Bisection。遞歸分群的思想是：先對整個數據集進行初次分群，然後對每個子簇再繼續應用分群算法，直到達到某個終止條件。
1. 基本思路
遞歸分群的過程可以大致分爲以下幾步：
    1. 初次分群：使用分群算法（如 KMeans、DBSCAN 等）對整個數據集進行初次分群。
    2. 遞歸分群：對于每一個簇，檢查該簇內部是否可以繼續進行分群。如果簇內部的數據具有進一步的分群特征（比如某些分群指標達到阈值），則對該簇進行再次分群。
    3. 終止條件：設定終止條件，如簇內的樣本數少于某個阈值，或聚類指標（如輪廓系數、均方誤差等）滿足要求，則不再進行遞歸。
2. 遞歸 KMeans 的實現
我們將使用遞歸的方法來應用 KMeans 分群，同時設定一個阈值，遞歸地繼續對每個簇進行分群，直到滿足終止條件。
遞歸 KMeans 的代碼實現

"""
from stUtil import rndrCode
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def rcrsvKMeans(data, max_depth=3, min_cluster_size=5, depth=0): # 遞歸 KMeans 分群的函數 通用分群
    # 終止條件：深度達到最大深度或數據量小于最小簇大小
    if depth >= max_depth or len(data) < min_cluster_size:
        return None

    # 使用 KMeans 進行分群
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(data)

    # 計算輪廓系數，判斷是否需要進一步分群
    輪廓係數 = silhouette_score(data, clusters)
    rndrCode(f"Depth: {depth}, Silhouette Score: {輪廓係數:.4f}")   #silhouette_avg

    if  輪廓係數< 0.5:  # 如果分群效果較差，則停止遞歸  silhouette_avg
        return None

    # 創建一個字典來存儲簇的遞歸結果
    次群 = {}

    for cluster_label in set(clusters):
        cluster_data = data[clusters == cluster_label]

        if len(cluster_data) > min_cluster_size:  # 僅對滿足條件的簇繼續遞歸分群
            rndrCode(f"Recursively clustering on cluster {cluster_label} at depth {depth}")
            次群[cluster_label] = rcrsvKMeans(cluster_data, max_depth=max_depth, depth=depth + 1)
        else:
            次群[cluster_label] = None

    return 次群

# 使用示例數據進行遞歸 KMeans 分群
"""
3. 解釋代碼邏輯
    1. 遞歸調用：recursive_kmeans 函數首先對傳入的數據集進行 KMeans 分群，然後檢查分群的效果（通過計算輪廓系數等聚類指標）。如果效果較好，則繼續對每個簇中的數據進行遞歸分群，直到達到最大深度或簇的大小不足以繼續分群。
    2. 終止條件：
        ◦ 遞歸深度達到 max_depth（即超過了遞歸允許的最大層級）。
        ◦ 當前簇的數據量小于 min_cluster_size（即該簇的數據量不足以進行進一步的分群）。
        ◦ 使用 Silhouette Score 判斷當前分群的質量，若該值過低，則停止遞歸。
    3. 簇的分群：對于每一個簇，如果該簇的數據量足夠大，且分群效果好，則遞歸地繼續對該簇進行分群。最終，返回所有簇的嵌套結構。
4. 結合其他分群算法
同樣的遞歸分群邏輯可以應用于其他分群算法，例如 DBSCAN、Agglomerative Clustering 等。我們可以通過調整算法的選擇和遞歸條件，實現對數據的靈活分群。
使用 DBSCAN 進行遞歸分群
from sklearn.cluster import DBSCAN

"""

def rcrsvDBSCAN(data, eps=0.5, min_samples=5, max_depth=3, min_cluster_size=5, depth=0):   #recursive_dbscan 遞歸 DBSCAN 分群的函數
    # 終止條件：深度達到最大深度或數據量小于最小簇大小
    if depth >= max_depth or len(data) < min_cluster_size:
        return None

    # 使用 DBSCAN 進行分群
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(data)

    # 檢查是否產生了多個簇
    unique_clusters = set(clusters)

    if len(unique_clusters) <= 1:  # 如果只有一個簇或者沒有簇，則停止遞歸
        return None

    次群 = {}

    for cluster_label in unique_clusters:
        if cluster_label != -1:  # -1 表示噪聲點，不進行遞歸
            cluster_data = data[clusters == cluster_label]

            if len(cluster_data) > min_cluster_size:  # 僅對滿足條件的簇繼續遞歸分群
                rndrCode(f"Recursively clustering on cluster {cluster_label} at depth {depth}")
                次群[cluster_label] = rcrsvDBSCAN(cluster_data, eps=eps, min_samples=min_samples, max_depth=max_depth, depth=depth + 1)
            else:
                次群[cluster_label] = None

    return 次群

# 使用示例數據進行遞歸 DBSCAN 分群
#5. 遞歸分群的可視化 在遞歸分群過程中，可以將每次分群的結果通過可視化的方式展示，這樣可以更直觀地了解遞歸過程中的簇划分情況。
# 可視化遞歸分群結果
def vis分群(data, clusters, depth=0):
  from sklearn.decomposition import PCA
  from matplotlib.pyplot import figure
  from seaborn import scatterplot
  from streamlit import pyplot
  pca = PCA(n_components=2)
  pca_data = pca.fit_transform(data)
  fig=figure(figsize=(10, 6))
  scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='viridis')
  pyplot(fig)
  #plt.title(f"Cluster Visualization at Depth {depth}")
  #plt.show()

"""
6. 優化遞歸分群的策略
    1. 參數調整：在遞歸過程中，可以動態調整分群算法的參數。例如，在更深層次的遞歸中，可以逐步減少 KMeans 的簇數量，或調整 DBSCAN 的 eps 和 min_samples 參數。
    2. 并行化遞歸：當數據集較大時，遞歸分群的計算可以進行并行化處理，尤其是在處理大規模數據集時，可以通過多線程或分布式計算來提升效率。
總結
使用遞歸方法進行通用分群，可以對數據集進行多層次、細粒度的聚類。通過遞歸調用不同的分群算法，我們可以實現對不同層次簇的精細化划分，并根據分群效果動態調整模型。最終，結合可視化技術，可以更好地理解數據的內部結構和遞歸分群的層級關系。
"""
