import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

embeddings = np.loadtxt('vec.csv', delimiter=',', skiprows=1)
print(np.std(embeddings[:, 0]))
# Plot the t-SNE results
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings)
# Create a DataFrame with the t-SNE results
tsne_df = pd.DataFrame(embeddings_tsne, columns=['Dim1', 'Dim2'])
tsne_df['label'] = df['label']
plt.figure(figsize=(10, 8))
sns.scatterplot(data=tsne_df, x='Dim1', y='Dim2', hue='label', palette='viridis', s=50, alpha=0.7)
plt.title('t-SNE of Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("sne.png")

# Perform PCA to reduce the dimensionality to 2
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)
# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(embeddings_pca, columns=['PC1', 'PC2'])
pca_df['label'] = df['label']
# Plot the PCA results
plt.figure(figsize=(10, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='label', palette='viridis', s=50, alpha=0.7)
plt.title('PCA of Embeddings')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("pca.png")
