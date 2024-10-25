import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def realizar_pca_y_visualizar(X_vectorized, y):
    # Aplicar PCA para reducir a 2 componentes principales
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_vectorized.toarray())

    # Visualizar los datos en 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.title('Visualización de Tweets usando PCA')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.colorbar(label='Sentimiento')
    plt.show()
