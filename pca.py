# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Step 2: Load wine dataset
wine = load_wine()
X = wine.data           # features
y = wine.target         # class labels

# Step 3: Scale the data (important before PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply PCA (reduce dimensions to 2)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Convert to DataFrame
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['target'] = y

# Step 6: Plot the PCA result
plt.figure(figsize=(7,5))
for cls in df_pca['target'].unique():
    subset = df_pca[df_pca['target'] == cls]
    plt.scatter(subset['PC1'], subset['PC2'], label=f"Class {cls}")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Wine Dataset")
plt.legend()
plt.show()