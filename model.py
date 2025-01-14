import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from data_analysis import realizar_pca_y_visualizar

# Cargar el dataset desde el archivo CSV descomprimido con codificación 'latin-1'
data = pd.read_csv('tweets.csv', encoding='latin-1', header=None)

# Renombrar las columnas para facilitar el trabajo
data.columns = ['polarity', 'id', 'date', 'query', 'user', 'tweet']

# Convertir las etiquetas de polaridad: 0 (negativo), 4 (positivo)
# Convertimos la polaridad 4 a 1 (positivo) y mantenemos la polaridad 0 (negativo)
X = data['tweet']
y = data['polarity'].apply(lambda x: 1 if x == 4 else 0)

# Después de cargar los datos
sample_size = 100000  # Ajusta este número según sea necesario
data = data.sample(n=sample_size, random_state=42)

# Preprocesamiento de texto y vectorización con TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)  # Reducir de 5000 a 1000
X_vectorized = vectorizer.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Guardar el modelo y el vectorizador en archivos .pkl
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Llamar a la función para realizar PCA y visualizar
graph_json = realizar_pca_y_visualizar(X_vectorized, y)

# Guardar los datos del gráfico en un archivo
with open('pca_graph.json', 'w') as f:
    f.write(graph_json)

print("Modelo, vectorizador y datos del gráfico guardados con éxito.")
