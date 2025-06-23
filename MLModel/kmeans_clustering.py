import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv('all_feature_importances.csv', index_col=0)

# 转置数据，使每一列成为一组数据
data = data.T

# 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 使用KMeans进行聚类，调整聚类数量和初始化方法
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(data_scaled)
labels = kmeans.labels_

# 使用PCA进行降维以便可视化，增加主成分数量
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)

# 可视化聚类结果
plt.figure(figsize=(10, 7))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=labels, cmap='viridis')
plt.title('KMeans Clustering of Feature Importances')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# 保存xy都小于0的数据，并包含第一列的表头信息
negative_data = principal_components[(principal_components[:, 0] < 0) & (principal_components[:, 1] < 0)]
negative_data_df = pd.DataFrame(negative_data, columns=['Principal Component 1', 'Principal Component 2'])
negative_data_df['Feature'] = data.index[(principal_components[:, 0] < 0) & (principal_components[:, 1] < 0)]
negative_data_df.to_csv('negative_data.csv', index=False)
