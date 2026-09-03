import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer


# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="TF-IDF Search",
    page_icon="🔎",
    layout="wide"
)


# ==========================================================
# ESTILOS
# ==========================================================

st.markdown("""
<style>

    /* Fondo */
    .stApp {
        background-color: #f7f9fc;
    }

    /* Encabezado */
    .titulo {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #1f2937;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .subtitulo {
        text-align: center;
        font-size: 18px;
        color: #667085;
        margin-bottom: 30px;
    }

    /* Tarjetas */
    .card {
        background-color: white;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e4e7ec;
        margin-bottom: 20px;
        box-shadow: 0px 3px 12px rgba(0, 0, 0, 0.05);
    }

    /* Resultado */
    .resultado {
        background-color: white;
        border-radius: 16px;
        padding: 28px;
        border: 1px solid #d9e2ec;
        margin-top: 20px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.06);
    }

    .resultado-titulo {
        color: #1f2937;
        font-size: 24px;
        font-weight: 700;
    }

    .relevancia {
        font-size: 32px;
        font-weight: 700;
    }

    /* Texto pequeño */
    .descripcion {
        color: #667085;
        font-size: 16px;
        line-height: 1.6;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #98a2b3;
        padding: 30px;
        margin-top: 30px;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================================
# ENCABEZADO
# ==========================================================

st.markdown(
    '<div class="titulo">🔎 TF-IDF Search</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitulo">'
    'Encuentra el documento más relacionado con tu pregunta'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("🛠️ Herramientas")

    st.write(
        "Configura tu búsqueda y administra la base de conocimiento."
    )

    st.divider()

    st.subheader("📚 ¿Qué utilizamos?")

    st.write("""
    **TF-IDF**

    Permite identificar la importancia de las palabras
    dentro de los documentos.

    **Stemming**

    Reduce las palabras a una forma básica.

    **Similitud coseno**

    Permite comparar la pregunta con los documentos
    y encontrar el más relacionado.
    """)

    st.divider()

    st.info(
        "💡 Cada salto de línea representa un documento independiente."
    )


# ==========================================================
# BASE DE CONOCIMIENTO
# ==========================================================

st.markdown(
    "### 📚 Base de conocimiento"
)

st.markdown(
    '<div class="descripcion">'
    'Escribe tus documentos. Cada línea será procesada como un documento independiente.'
    '</div>',
    unsafe_allow_html=True
)


default_docs = """La fotosíntesis es el proceso mediante el cual las plantas transforman la energía luminosa en energía química, liberando oxígeno al ambiente.
El Coliseo de Roma fue construido en el siglo I y es un anfiteatro famoso por su arquitectura y sus túneles.
El telescopio espacial James Webb utiliza sensores infrarrojos para observar galaxias distantes y estudiar el universo.
La cocina japonesa tradicional utiliza ingredientes de temporada, sabores equilibrados y técnicas precisas de preparación.
Los estudiantes utilizan diferentes herramientas tecnológicas para desarrollar sus proyectos académicos.
La inteligencia artificial permite analizar grandes cantidades de información y encontrar patrones en los datos.
La programación permite desarrollar aplicaciones y soluciones para diferentes necesidades.
La educación universitaria combina conocimientos teóricos y prácticos para preparar a los estudiantes para el mundo profesional."""


# ==========================================================
# CARGAR ARCHIVO TXT
# ==========================================================

archivo = st.file_uploader(
    "📤 Sube tu base de conocimiento (.txt)",
    type=["txt"]
)


if archivo is not None:

    try:

        contenido = archivo.read().decode("utf-8")

        documents = [
            d.strip()
            for d in contenido.split("\n")
            if d.strip()
        ]

        st.success(
            f"✅ Base cargada correctamente: {len(documents)} documentos."
        )

    except Exception:

        st.error(
            "No fue posible leer el archivo. "
            "Verifica que sea un archivo TXT en formato UTF-8."
        )

        documents = []

else:

    documents = [
        d.strip()
        for d in default_docs.split("\n")
        if d.strip()
    ]

    st.info(
        "Estás utilizando la base de conocimiento de ejemplo."
    )


# ==========================================================
# MOSTRAR BASE DE CONOCIMIENTO
# ==========================================================

with st.expander("📄 Ver documentos de la base"):

    for i, document in enumerate(documents):

        st.write(
            f"**Documento {i + 1}:** {document}"
        )


# ==========================================================
# PREGUNTA
# ==========================================================

st.markdown(
    "### 🔍 Búsqueda inteligente"
)

st.markdown(
    '<div class="descripcion">'
    'Escribe una pregunta y el sistema encontrará el documento más relevante.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================================
# PREGUNTAS SUGERIDAS
# ==========================================================

preguntas = [
    "¿Cómo funciona la fotosíntesis?",
    "¿Cuándo se construyó el Coliseo?",
    "¿Qué estudia el telescopio James Webb?",
    "¿Qué caracteriza a la cocina japonesa?",
    "¿Qué permite hacer la inteligencia artificial?",
    "¿Para qué sirve la programación?"
]


if "pregunta_seleccionada" not in st.session_state:

    st.session_state.pregunta_seleccionada = ""


col1, col2 = st.columns([3, 1])


with col1:

    question = st.text_input(
        "Haz tu pregunta:",
        value=st.session_state.pregunta_seleccionada,
        placeholder="Ejemplo: ¿Qué estudia el telescopio James Webb?",
        key="pregunta"
    )


with col2:

    st.markdown("#### 💡 Preguntas sugeridas")

    for i, pregunta in enumerate(preguntas):

        if st.button(
            pregunta,
            key=f"sugerida_{i}",
            use_container_width=True
        ):

            st.session_state.pregunta_seleccionada = pregunta

            st.rerun()


# ==========================================================
# BOTÓN DE BÚSQUEDA
# ==========================================================

buscar = st.button(
    "🔎 Ejecutar búsqueda",
    type="primary",
    use_container_width=True
)


# ==========================================================
# FUNCIÓN DE TOKENIZACIÓN Y STEMMING
# ==========================================================

stemmer = SnowballStemmer("spanish")


def tokenize_and_stem(text):

    # Convertir a minúsculas

    text = text.lower()

    # Mantener letras españolas

    text = re.sub(
        r'[^a-záéíóúüñ\s]',
        ' ',
        text
    )

    # Separar palabras

    tokens = [
        t
        for t in text.split()
        if len(t) > 1
    ]

    # Aplicar stemming

    stems = [
        stemmer.stem(t)
        for t in tokens
    ]

    return stems


# ==========================================================
# PROCESAMIENTO
# ==========================================================

if buscar:

    if len(documents) < 1:

        st.error(
            "⚠️ Debes ingresar al menos un documento."
        )

    elif not question.strip():

        st.error(
            "⚠️ Escribe una pregunta para realizar la búsqueda."
        )

    else:

        # ==================================================
        # CREAR VECTORIZADOR TF-IDF
        # ==================================================

        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            token_pattern=None,
            min_df=1
        )


        # ==================================================
        # CREAR MATRIZ TF-IDF
        # ==================================================

        X = vectorizer.fit_transform(
            documents
        )


        # ==================================================
        # TRANSFORMAR LA PREGUNTA
        # ==================================================

        question_vec = vectorizer.transform(
            [question]
        )


        # ==================================================
        # SIMILITUD COSENO
        # ==================================================

        similarities = cosine_similarity(
            question_vec,
            X
        ).flatten()


        # ==================================================
        # DOCUMENTO MÁS RELEVANTE
        # ==================================================

        best_idx = similarities.argmax()

        best_doc = documents[best_idx]

        best_score = similarities[best_idx]


        # ==================================================
        # RESULTADO
        # ==================================================

        st.markdown(
            '<div class="resultado">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="resultado-titulo">'
            '🎯 Resultado más relevante'
            '</div>',
            unsafe_allow_html=True
        )

        st.write("")

        st.write(
            f"**Tu pregunta:** {question}"
        )

        st.write(
            f"**Documento seleccionado:** Documento {best_idx + 1}"
        )

        st.success(
            best_doc
        )

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "📈 Similitud",
                f"{best_score:.3f}"
            )


        with col2:

            st.metric(
                "📊 Relevancia",
                f"{best_score * 100:.1f}%"
            )


        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ==================================================
        # TABS DE ANÁLISIS
        # ==================================================

        tab1, tab2, tab3 = st.tabs([
            "📊 Ranking",
            "🔢 Matriz TF-IDF",
            "✂️ Stemming"
        ])


        # ==================================================
        # RANKING
        # ==================================================

        with tab1:

            st.markdown(
                "### 📈 Documentos ordenados por relevancia"
            )

            sim_df = pd.DataFrame({

                "Documento": [
                    f"Documento {i + 1}"
                    for i in range(len(documents))
                ],

                "Texto": documents,

                "Similitud": similarities

            })


            sim_df = sim_df.sort_values(
                "Similitud",
                ascending=False
            )


            sim_df["Similitud"] = sim_df[
                "Similitud"
            ].round(3)


            sim_df["Relevancia (%)"] = (
                sim_df["Similitud"] * 100
            ).round(1)


            st.dataframe(
                sim_df,
                use_container_width=True,
                hide_index=True
            )


        # ==================================================
        # MATRIZ TF-IDF
        # ==================================================

        with tab2:

            st.markdown(
                "### 🔢 Matriz TF-IDF"
            )

            st.write(
                "Cada fila representa un documento y cada columna "
                "representa una palabra procesada."
            )


            df_tfidf = pd.DataFrame(

                X.toarray(),

                columns=vectorizer.get_feature_names_out(),

                index=[
                    f"Doc {i + 1}"
                    for i in range(len(documents))
                ]

            )


            st.dataframe(
                df_tfidf.round(3),
                use_container_width=True
            )


        # ==================================================
        # STEMMING
        # ==================================================

        with tab3:

            st.markdown(
                "### ✂️ Palabras procesadas"
            )

            st.write(
                "El stemming transforma las palabras a una forma "
                "reducida para facilitar la comparación."
            )


            # Stems de la pregunta

            q_stems = tokenize_and_stem(
                question
            )


            # Vocabulario

            vocab = (
                vectorizer
                .get_feature_names_out()
            )


            # Stems que aparecen en el documento elegido

            matched = [

                stem

                for stem in q_stems

                if (
                    stem in vocab
                    and df_tfidf.iloc[
                        best_idx
                    ].get(stem, 0) > 0
                )

            ]


            st.write(
                "**Stems de la pregunta:**"
            )

            st.write(
                q_stems
            )


            st.write(
                "**Stems encontrados en el documento seleccionado:**"
            )

            st.write(
                matched
            )


        # ==================================================
        # INFORMACIÓN DEL PROCESO
        # ==================================================

        with st.expander(
            "🧠 Ver cómo se realizó la búsqueda"
        ):

            st.write("""
            **1. Tokenización**

            El texto se divide en palabras.

            **2. Stemming**

            Las palabras se reducen a una forma básica.

            **3. TF-IDF**

            Cada documento se convierte en un vector numérico.

            **4. Vectorización de la pregunta**

            La pregunta se transforma utilizando el mismo vocabulario.

            **5. Similitud coseno**

            Se compara la pregunta con cada documento.

            **6. Ranking**

            Los documentos se ordenan según su similitud.

            **7. Resultado**

            Se selecciona el documento con mayor similitud.
            """)


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("""
<div class="footer">

🔎 <strong>TF-IDF Search</strong><br>

Búsqueda de información mediante procesamiento de lenguaje natural.

</div>
""", unsafe_allow_html=True)
