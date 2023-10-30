import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv', header=None)
plt.scatter(data[0], data[1])
plt.title("Scatter Plot for the Original Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
bkmp = KMeans(n_clusters=3, init='k-means++', random_state=42, algorithm='elkan')
bkmp.fit(data)
labels = bkmp.labels_
plt.scatter(data[0], data[1], c = labels)
plt.title("Scatter Plot with Labeled Clusters")
plt.scatter(bkmp.cluster_centers_[0][0], bkmp.cluster_centers_[0][1], c = 'r', marker='x')
plt.scatter(bkmp.cluster_centers_[1][0], bkmp.cluster_centers_[1][1], c = 'g', marker='x')
plt.scatter(bkmp.cluster_centers_[2][0], bkmp.cluster_centers_[2][1], c = 'b', marker='x')

plt.xlabel("X")
plt.ylabel("Y")
plt.show()
print(len(labels))
df = pd.DataFrame(data = labels)
df.to_csv(r'final_labels.txt', header=None, index=None, sep=' ')