
import re
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import SnowballStemmer


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

st.set_page_config(
    page_title="NARANJA | Buscador inteligente",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# ESTILOS PERSONALIZADOS
# ==========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #FFF8F1;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background-color: #FFEBDD;
        border-right: 1px solid #F3D3BD;
    }

    .hero {
        background: linear-gradient(135deg, #F58A45, #D95522);
        padding: 34px 30px;
        border-radius: 24px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 40px;
        margin: 0;
    }

    .hero p {
        color: white;
        font-size: 17px;
        margin-top: 10px;
        margin-bottom: 0;
    }

    .section-label {
        color: #B74B20;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #F1D8C7;
        border-radius: 17px;
        padding: 20px;
        margin-bottom: 12px;
    }

    .info-card h3 {
        color: #873C1C;
        margin-top: 0;
    }

    .muted {
        color: #77675D;
        line-height: 1.7;
    }

    .result-card {
        background-color: white;
        border: 1px solid #F0D4C1;
        border-left: 6px solid #F07836;
        border-radius: 16px;
        padding: 23px;
        margin: 14px 0;
    }

    .result-card h3 {
        color: #8C451D;
        margin-top: 0;
    }

    div.stButton > button {
        border-radius: 11px;
        font-weight: 600;
        min-height: 42px;
    }

    div.stButton > button[kind="primary"] {
        background-color: #F07836;
        border-color: #F07836;
        color: white;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #D95D23;
        border-color: #D95D23;
        color: white;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #F1D8C7;
        border-radius: 15px;
        padding: 15px;
    }

    .footer {
        text-align: center;
        color: #8F8177;
        padding: 24px;
        margin-top: 35px;
        border-top: 1px solid #EAD6C8;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# FUNCIONES DE PROCESAMIENTO
# ==========================================================

stemmer = SnowballStemmer("spanish")


def tokenize_and_stem(text):
    """Normaliza, separa y reduce las palabras en español."""

    text = text.lower()

    # Conserva letras españolas y elimina signos y números.
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)

    tokens = [
        token
        for token in text.split()
        if len(token) > 1
    ]

    return [stemmer.stem(token) for token in tokens]


def split_documents(text):
    """Cada línea no vacía representa un documento."""

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def create_vectorizer():
    """Crea el vectorizador TF-IDF con stemming."""

    return TfidfVectorizer(
        tokenizer=tokenize_and_stem,
        token_pattern=None,
        lowercase=False,
        min_df=1
    )


# ==========================================================
# BASE DE CONOCIMIENTO DE EJEMPLO
# ==========================================================

DEFAULT_DOCS = """La fotosíntesis es el proceso mediante el cual las plantas transforman la energía luminosa en energía química y liberan oxígeno al ambiente.
El Coliseo de Roma fue construido en el siglo I y es un anfiteatro famoso por su arquitectura y sus túneles.
El telescopio espacial James Webb utiliza sensores infrarrojos para observar galaxias distantes y estudiar el universo.
La cocina japonesa tradicional utiliza ingredientes de temporada, sabores equilibrados y técnicas precisas de preparación.
Los estudiantes utilizan diferentes herramientas tecnológicas para desarrollar sus proyectos académicos.
La inteligencia artificial permite analizar grandes cantidades de información y encontrar patrones en los datos.
La programación permite desarrollar aplicaciones y soluciones para diferentes necesidades.
La educación universitaria combina conocimientos teóricos y prácticos para preparar a los estudiantes para el mundo profesional."""


PREGUNTAS = [
    "¿Cómo funciona la fotosíntesis?",
    "¿Cuándo se construyó el Coliseo?",
    "¿Qué estudia el telescopio James Webb?",
    "¿Qué caracteriza a la cocina japonesa?",
    "¿Qué permite hacer la inteligencia artificial?",
    "¿Para qué sirve la programación?"
]


# ==========================================================
# ENCABEZADO
# ==========================================================

st.markdown("""
<div class="hero">
    <div class="section-label" style="color:#FFE1CA;">
        PROCESAMIENTO DE LENGUAJE NATURAL
    </div>
    <h1>🔎 NARANJA Search</h1>
    <p>
        Encuentra conexiones entre tus preguntas y tus documentos.
        Busca información de forma inteligente con TF-IDF.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# BARRA LATERAL
# ==========================================================

with st.sidebar:

    st.markdown("## 🍊 NARANJA Search")

    st.write(
        "Tu espacio para explorar información y encontrar "
        "los documentos más relacionados con cada pregunta."
    )

    st.divider()

    st.markdown("### 🧠 ¿Cómo funciona?")

    st.markdown("""
    **1. TF-IDF**

    Calcula la importancia de las palabras.

    **2. Stemming**

    Reduce las palabras a sus raíces aproximadas.

    **3. Similitud coseno**

    Compara la pregunta con cada documento.

    **4. Ranking**

    Ordena los documentos por similitud.
    """)

    st.divider()

    st.markdown("### 📌 Recomendaciones")

    st.caption(
        "Cada línea del archivo TXT se interpreta como un documento "
        "independiente. Escribe preguntas con palabras relacionadas "
        "con la información que buscas."
    )

    st.divider()

    st.caption("Proyecto académico · Procesamiento de Lenguaje Natural")


# ==========================================================
# CARGA DE DOCUMENTOS
# ==========================================================

st.markdown('<p class="section-label">01 — EXPLORA</p>',
            unsafe_allow_html=True)

st.header("📚 Tu base de conocimiento")

st.markdown(
    '<p class="muted">Carga tu propia colección de textos o '
    'comienza con los documentos de ejemplo.</p>',
    unsafe_allow_html=True
)

archivo = st.file_uploader(
    "Importa tus documentos (.txt)",
    type=["txt"],
    help="Cada línea del archivo representa un documento."
)

if archivo is not None:

    try:
        contenido = archivo.getvalue().decode("utf-8-sig")
        documents = split_documents(contenido)

        if not documents:
            st.warning(
                "El archivo no contiene líneas con texto. "
                "Prueba con otro archivo."
            )

    except UnicodeDecodeError:
        st.error(
            "No se pudo leer el archivo. Guárdalo como TXT "
            "codificado en UTF-8."
        )
        documents = []

else:
    documents = split_documents(DEFAULT_DOCS)
    st.info("Estás utilizando la base de conocimiento de ejemplo.")

# Eliminar documentos que no contengan palabras procesables.
documents_validos = [
    doc for doc in documents
    if tokenize_and_stem(doc)
]

if documents and not documents_validos:
    st.warning(
        "No encontramos palabras que puedan procesarse en los "
        "documentos. Revisa el contenido del archivo."
    )

documents = documents_validos

col_a, col_b = st.columns(2)

with col_a:
    st.metric("📄 Documentos disponibles", len(documents))

with col_b:
    st.metric(
        "🔤 Palabras únicas",
        len({
            palabra
            for doc in documents
            for palabra in tokenize_and_stem(doc)
        })
    )

with st.expander("👀 Explorar documentos"):

    if documents:
        for i, doc in enumerate(documents):
            st.markdown(f"**Documento {i + 1}**")
            st.write(doc)
            st.divider()
    else:
        st.write("No hay documentos disponibles.")


# ==========================================================
# BÚSQUEDA INTELIGENTE
# ==========================================================

st.markdown("---")

st.markdown(
    '<p class="section-label">02 — BUSCA</p>',
    unsafe_allow_html=True
)

st.header("🔍 ¿Qué información necesitas?")

st.write(
    "Escribe una pregunta o selecciona un ejemplo. "
    "El sistema comparará tu consulta con los documentos."
)

# Inicializar estado antes de mostrar el campo de texto.
if "pregunta" not in st.session_state:
    st.session_state["pregunta"] = ""

if "ultima_busqueda" not in st.session_state:
    st.session_state["ultima_busqueda"] = None

with st.expander("💡 Probar preguntas sugeridas"):

    pregunta_ejemplo = st.selectbox(
        "Selecciona una pregunta",
        PREGUNTAS,
        key="selector_pregunta"
    )

    if st.button(
        "Usar esta pregunta",
        key="usar_pregunta",
        use_container_width=True
    ):
        st.session_state["pregunta"] = pregunta_ejemplo
        st.rerun()

question = st.text_input(
    "Tu pregunta",
    key="pregunta",
    placeholder="Ej.: ¿Qué permite hacer la inteligencia artificial?"
)

buscar = st.button(
    "🔎 Buscar documentos",
    type="primary",
    use_container_width=True
)


# ==========================================================
# PROCESAMIENTO TF-IDF
# ==========================================================

if buscar:

    if not documents:
        st.error(
            "No hay documentos válidos. Carga un archivo TXT "
            "con contenido o utiliza una base de ejemplo."
        )

    elif not question.strip():
        st.warning("Escribe una pregunta antes de buscar.")

    else:
        try:
            vectorizer = create_vectorizer()

            # Crear la matriz TF-IDF de los documentos.
            X = vectorizer.fit_transform(documents)

            # Convertir la pregunta al mismo espacio vectorial.
            question_vec = vectorizer.transform([question])

            # Comparar la pregunta con todos los documentos.
            similarities = cosine_similarity(
                question_vec,
                X
            ).flatten()

            # Ordenar de mayor a menor similitud.
            ranking = similarities.argsort()[::-1]

            st.session_state["ultima_busqueda"] = {
                "question": question,
                "documents": documents,
                "similarities": similarities.tolist(),
                "ranking": ranking.tolist(),
                "matrix": X.toarray().tolist(),
                "vocabulary": vectorizer.get_feature_names_out().tolist()
            }

        except ValueError as error:
            st.error(
                "No fue posible procesar los documentos. "
                "Revisa que contengan texto válido."
            )
            st.caption(f"Detalle técnico: {error}")

        except Exception as error:
            st.error("Ocurrió un error durante la búsqueda.")
            st.caption(f"Detalle técnico: {error}")


# ==========================================================
# PRESENTACIÓN DE RESULTADOS
# ==========================================================

data = st.session_state["ultima_busqueda"]

if data is not None:

    st.markdown("---")

    st.markdown(
        '<p class="section-label">03 — DESCUBRE</p>',
        unsafe_allow_html=True
    )

    st.header("🎯 Resultados de tu búsqueda")

    st.write(f"**Pregunta:** {data['question']}")

    similarities = data["similarities"]
    ranking = data["ranking"]
    docs = data["documents"]

    best_idx = ranking[0]
    best_score = similarities[best_idx]

    if best_score <= 0:

        st.warning(
            "No encontramos coincidencias entre las palabras "
            "de tu pregunta y los documentos. Prueba con otros "
            "términos o revisa la base de conocimiento."
        )

    else:

        st.markdown(
            f"""
            <div class="result-card">
                <h3>🏆 Documento más relacionado</h3>
                <p><strong>Documento {best_idx + 1}</strong></p>
                <p>{docs[best_idx]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        met1, met2, met3 = st.columns(3)

        with met1:
            st.metric(
                "📈 Similitud coseno",
                f"{best_score:.3f}"
            )

        with met2:
            st.metric(
                "📊 Similitud relativa",
                f"{best_score * 100:.1f}%"
            )

        with met3:
            st.metric(
                "📚 Documentos comparados",
                len(docs)
            )

    # ------------------------------------------------------
    # PESTAÑAS DE ANÁLISIS
    # ------------------------------------------------------

    tab1, tab2, tab3 = st.tabs([
        "📊 Ranking",
        "🔢 Matriz TF-IDF",
        "✂️ Stemming"
    ])

    # ======================================================
    # RANKING
    # ======================================================

    with tab1:

        st.subheader("Documentos ordenados por similitud")

        ranking_df = pd.DataFrame([
            {
                "Posición": posicion + 1,
                "Documento": f"Documento {idx + 1}",
                "Contenido": docs[idx],
                "Similitud": similarities[idx],
                "Similitud (%)": similarities[idx] * 100
            }
            for posicion, idx in enumerate(ranking)
        ])

        ranking_df["Similitud"] = ranking_df["Similitud"].round(3)
        ranking_df["Similitud (%)"] = (
            ranking_df["Similitud (%)"].round(1)
        )

        st.dataframe(
            ranking_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Similitud": st.column_config.NumberColumn(
                    format="%.3f"
                ),
                "Similitud (%)": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=100,
                    format="%.1f%%"
                )
            }
        )

        st.caption(
            "Una similitud de cero significa que la consulta "
            "no comparte características léxicas ponderadas "
            "con el documento en este modelo."
        )

    # ======================================================
    # MATRIZ TF-IDF
    # ======================================================

    with tab2:

        st.subheader("Matriz TF-IDF")

        st.write(
            "Cada fila representa un documento y cada columna "
            "representa un término procesado. Los valores indican "
            "el peso TF-IDF de cada término."
        )

        vocab = data["vocabulary"]
        matriz = data["matrix"]

        if vocab:
            df_tfidf = pd.DataFrame(
                matriz,
                columns=vocab,
                index=[
                    f"Documento {i + 1}"
                    for i in range(len(docs))
                ]
            )

            st.dataframe(
                df_tfidf.round(3),
                use_container_width=True
            )

            st.download_button(
                "⬇️ Descargar matriz CSV",
                data=df_tfidf.to_csv().encode("utf-8-sig"),
                file_name="matriz_tfidf.csv",
                mime="text/csv"
            )

        else:
            st.info("No hay términos para mostrar.")

    # ======================================================
    # STEMMING
    # ======================================================

    with tab3:

        st.subheader("Análisis de palabras")

        q_stems = tokenize_and_stem(data["question"])

        st.markdown("**Palabras procesadas de la pregunta**")

        st.write(q_stems)

        vocabulario = set(data["vocabulary"])

        st.markdown("**Términos de la pregunta presentes en el vocabulario**")

        coincidencias = [
            stem for stem in q_stems
            if stem in vocabulario
        ]

        if coincidencias:
            st.write(coincidencias)
        else:
            st.info(
                "Ningún término procesado de la pregunta aparece "
                "en el vocabulario de los documentos."
            )

        st.markdown("**Comparación con el documento más relevante**")

        if best_score > 0:

            pesos = data["matrix"][best_idx]

            terminos_documento = {
                termino
                for termino, peso in zip(vocab, pesos)
                if peso > 0
            }

            coincidencias_doc = [
                stem for stem in q_stems
                if stem in terminos_documento
            ]

            st.write(coincidencias_doc)

        else:
            st.write(
                "No hay coincidencias ponderadas con el documento "
                "más relevante."
            )

    # ======================================================
    # EXPLICACIÓN
    # ======================================================

    with st.expander("🧠 ¿Cómo funciona esta búsqueda?"):

        st.markdown("""
        **1. Tokenización:** el texto se separa en palabras.

        **2. Stemming:** las palabras se reducen a raíces
        aproximadas para comparar formas relacionadas.

        **3. TF-IDF:** se calcula un peso para cada término
        según su presencia en los documentos y su frecuencia
        en la colección.

        **4. Vectorización:** la pregunta se convierte en un
        vector utilizando el vocabulario de los documentos.

        **5. Similitud coseno:** se compara el vector de la
        pregunta con cada vector documental.

        **6. Ranking:** se ordenan los documentos de mayor
        a menor similitud.

        **Importante:** TF-IDF compara características léxicas.
        No comprende el significado como un modelo semántico
        avanzado, por lo que una pregunta formulada con palabras
        distintas a las del documento puede obtener una similitud
        baja o igual a cero.
        """)


# ==========================================================
# PIE DE PÁGINA
# ==========================================================

st.markdown("""
<div class="footer">
    <strong>🍊 NARANJA Search</strong>
    <p>Explora información. Encuentra conexiones. Aprende con datos.</p>
    <small>Procesamiento de Lenguaje Natural · TF-IDF</small>
</div>
""", unsafe_allow_html=True)
