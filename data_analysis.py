import pandas as pd
from sklearn.decomposition import TruncatedSVD
import plotly.express as px
import json
import plotly

def realizar_pca_y_visualizar(X_vectorized, y):
    # Usar TruncatedSVD en lugar de PCA para manejar matrices dispersas
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_pca = svd.fit_transform(X_vectorized)

    # Crear el gráfico con Plotly
    df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    df['Sentiment'] = ['Positive' if s == 1 else 'Negative' for s in y]
    fig = px.scatter(df, x='PC1', y='PC2', color='Sentiment',
                     title='Visualización de Tweets usando SVD')

    # Convertir la figura a JSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON
