# 🔎 TF-IDF Search

Aplicación desarrollada en Streamlit para realizar búsquedas
dentro de una base de documentos utilizando técnicas de
Procesamiento de Lenguaje Natural.

## ✨ Funcionalidades

- 📚 Carga de documentos en formato TXT
- 🔎 Búsqueda mediante preguntas
- ✂️ Stemming en español
- 📊 Vectorización TF-IDF
- 🎯 Similitud coseno
- 📈 Ranking de documentos
- 🔢 Visualización de la matriz TF-IDF
- 💡 Preguntas sugeridas
- 🧠 Visualización del procesamiento de texto

## 🧰 Tecnologías

- Python
- Streamlit
- Scikit-learn
- Pandas
- NLTK

## ⚙️ Funcionamiento

La aplicación procesa cada línea de texto como un documento
independiente.

La pregunta del usuario se transforma en un vector TF-IDF y
se compara con los documentos mediante similitud coseno.

El documento con mayor similitud se presenta como el resultado
más relevante.

## 🚀 Ejecutar la aplicación

Instalar las dependencias:

```bash
pip install -r requirements.txt
