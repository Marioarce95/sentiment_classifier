from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
import json

app = Flask(__name__)

# Cargar el modelo y el vectorizador desde los archivos .pkl
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Cargar los datos del gráfico
with open('pca_graph.json', 'r') as f:
    graph_json = f.read()

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Limpieza básica del tweet
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)  # Elimina URLs y menciones
    text = text.lower()  # Convierte a minúsculas
    text = re.sub(r'\W', ' ', text)  # Elimina caracteres especiales
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        tweet = request.form['tweet']
        cleaned_tweet = clean_text(tweet)
        vectorized_tweet = vectorizer.transform([cleaned_tweet])
        prediction = model.predict(vectorized_tweet)[0]
        sentiment = 'Positivo' if prediction == 1 else 'Negativo'
        return render_template('index.html', sentiment=sentiment, tweet=tweet, graph_json=graph_json)
    return render_template('index.html', sentiment=None, graph_json=graph_json)

if __name__ == '__main__':
    app.run(debug=True)
