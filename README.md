# Clasificador de Sentimientos de Tweets

Esta aplicación es un clasificador de sentimientos para tweets en inglés. Utiliza técnicas de procesamiento de lenguaje natural y aprendizaje automático para determinar si un tweet tiene un sentimiento positivo o negativo.

## Características

- Clasificación de sentimientos de tweets en tiempo real
- Visualización de datos usando PCA (Análisis de Componentes Principales)
- Interfaz web simple para ingresar tweets y ver resultados

## Requisitos

- Python 3.7+
- pip (gestor de paquetes de Python)

## Instalación

1. Clona este repositorio:
   ```
   git clone https://github.com/tu-usuario/sentiment_classifier.git
   cd sentiment_classifier
   ```

2. Crea un entorno virtual y actívalo:
   ```
   python -m venv venv
   source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
   ```

3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Preparación de datos y entrenamiento del modelo

1. Asegúrate de tener el archivo `tweets.csv` en el directorio raíz del proyecto.

2. Ejecuta el script para entrenar el modelo:
   ```
   python model.py
   ```

   Esto creará los archivos `sentiment_model.pkl`, `tfidf_vectorizer.pkl` y `pca_graph.json`.

## Ejecución de la aplicación

1. Una vez que el modelo esté entrenado, puedes iniciar la aplicación Flask:
   ```
   python app.py
   ```

2. Abre un navegador y ve a `http://127.0.0.1:5000` para usar la aplicación.

## Uso

1. En la interfaz web, ingresa un tweet en inglés en el campo de texto.
2. Haz clic en "Analizar Sentimiento".
3. La aplicación mostrará si el sentimiento del tweet es positivo o negativo.
4. También podrás ver una visualización de PCA de los datos de entrenamiento.

## Estructura del proyecto

- `model.py`: Script para entrenar el modelo y generar los archivos necesarios.
- `data_analysis.py`: Contiene funciones para el análisis de datos y visualización.
- `app.py`: La aplicación Flask que sirve la interfaz web.
- `requirements.txt`: Lista de dependencias del proyecto.
- `templates/`: Directorio que contiene las plantillas HTML.

## Notas

- Asegúrate de tener suficiente espacio en disco y memoria RAM para procesar el conjunto de datos completo.
- El entrenamiento del modelo puede llevar tiempo dependiendo del tamaño del conjunto de datos y las capacidades de tu máquina.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de hacer un pull request.

## Licencia

[MIT License](https://opensource.org/licenses/MIT)
