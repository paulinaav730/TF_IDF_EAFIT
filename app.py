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
    page_title="EAFITRESPONDE",
    page_icon="💬",
    layout="wide"
)

# ==========================================================
# ESTILO VISUAL: AZULES CLAROS
# ==========================================================
st.markdown("""
<style>
    :root {
        --azul-oscuro: #174A7E;
        --azul: #3D83C6;
        --azul-medio: #78B7E5;
        --azul-claro: #DFF1FF;
        --fondo: #F4FAFF;
        --texto: #183B56;
    }

    .stApp {
        background: linear-gradient(180deg, #F4FAFF 0%, #FFFFFF 65%);
        color: var(--texto);
    }

    [data-testid="stHeader"] {
        background-color: rgba(244, 250, 255, 0.85);
    }

    .titulo {
        text-align: center;
        font-size: clamp(2.2rem, 5vw, 3.2rem);
        font-weight: 800;
        letter-spacing: -1.2px;
        color: var(--azul-oscuro);
        margin: 12px 0 4px 0;
    }

    .marca {
        color: var(--azul);
    }

    .subtitulo {
        text-align: center;
        font-size: 1.08rem;
        color: #54718A;
        margin: 0 auto 28px auto;
        max-width: 760px;
    }

    .separador {
        height: 4px;
        width: 90px;
        background: linear-gradient(90deg, #78B7E5, #DFF1FF);
        border-radius: 20px;
        margin: 12px auto 22px auto;
    }

    .descripcion {
        color: #54718A;
        font-size: 1rem;
        line-height: 1.65;
    }

    .resultado {
        background: linear-gradient(135deg, #FFFFFF 0%, #EAF6FF 100%);
        border: 1px solid #B9DDF6;
        border-radius: 18px;
        padding: 24px;
        margin-top: 18px;
        box-shadow: 0 8px 24px rgba(44, 112, 166, 0.08);
    }

    .resultado-titulo {
        color: #174A7E;
        font-size: 1.55rem;
        font-weight: 750;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #CDE7FA;
        border-radius: 14px;
        padding: 14px 16px;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #347FBD, #65A9DF);
        color: white;
        border: 0;
        border-radius: 12px;
        font-weight: 700;
        min-height: 46px;
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #24679F, #4B96D1);
        border: 0;
        box-shadow: 0 5px 14px rgba(52, 127, 189, 0.2);
    }

    div.stButton > button {
        border-radius: 10px;
        border-color: #B9DDF6;
    }

    div.stButton > button:hover {
        border-color: #5D9FD3;
        color: #174A7E;
    }

    [data-testid="stFileUploader"] {
        background-color: #FFFFFF;
        border: 1px dashed #9CCBEF;
        border-radius: 14px;
        padding: 10px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EAF6FF 0%, #F7FBFF 100%);
        border-right: 1px solid #D4EAFB;
    }

    .footer {
        text-align: center;
        color: #6C8CA5;
        padding: 28px 12px 12px 12px;
        margin-top: 30px;
        font-size: 0.92rem;
    }

    .footer strong {
        color: #174A7E;
    }

    h1, h2, h3 {
        color: #174A7E;
    }

    [data-testid="stExpander"] {
        border: 1px solid #D4EAFB;
        border-radius: 12px;
        background-color: rgba(255,255,255,0.8);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# ENCABEZADO
# ==========================================================
st.markdown(
    '<div class="titulo">💬 <span class="marca">EAFIT</span>RESPONDE</div>'
    '<div class="separador"></div>'
    '<div class="subtitulo">Tu buscador inteligente de información. '
    'Haz una pregunta y encuentra el documento más relacionado.</div>',
    unsafe_allow_html=True
)

# ==========================================================
# BARRA LATERAL
# ==========================================================
with st.sidebar:
    st.header("💡 EAFITRESPONDE")
    st.write("Consulta información de una base de documentos usando procesamiento de lenguaje natural.")
    st.divider()
    st.subheader("🧠 ¿Cómo funciona?")
    st.markdown("""
    **TF-IDF**  
    Calcula la importancia de las palabras en los documentos.

    **Stemming**  
    Reduce las palabras a una forma básica.

    **Similitud coseno**  
    Compara la pregunta con los documentos para encontrar los más relacionados.
    """)
    st.divider()
    st.info("Cada línea de tu archivo TXT se interpreta como un documento independiente.")

# ==========================================================
# BASE DE CONOCIMIENTO
# ==========================================================
st.markdown("### 📚 Base de conocimiento")
st.markdown(
    '<div class="descripcion">Carga un archivo TXT o utiliza los documentos de ejemplo. '
    'Cada línea será procesada como un documento independiente.</div>',
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

archivo = st.file_uploader(
    "📤 Sube tu base de conocimiento (.txt)",
    type=["txt"]
)

if archivo is not None:
    try:
        contenido = archivo.read().decode("utf-8")
        documents = [d.strip() for d in contenido.splitlines() if d.strip()]
        st.success(f"Base cargada correctamente: {len(documents)} documentos.")
    except Exception:
        st.error("No fue posible leer el archivo. Verifica que sea un TXT en formato UTF-8.")
        documents = []
else:
    documents = [d.strip() for d in default_docs.splitlines() if d.strip()]
    st.info("Estás utilizando la base de conocimiento de ejemplo.")

with st.expander("📄 Ver documentos de la base"):
    for i, document in enumerate(documents, start=1):
        st.write(f"**Documento {i}:** {document}")

# ==========================================================
# PREGUNTA Y SUGERENCIAS
# ==========================================================
st.markdown("### 🔎 ¿Qué quieres saber?")
st.markdown(
    '<div class="descripcion">Escribe una pregunta o elige una de las sugerencias '
    'para buscar el documento más relevante.</div>',
    unsafe_allow_html=True
)

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
    st.markdown("#### 💡 Prueba con una pregunta")
    for i, pregunta in enumerate(preguntas):
        if st.button(pregunta, key=f"sugerida_{i}", use_container_width=True):
            st.session_state.pregunta_seleccionada = pregunta
            st.rerun()

buscar = st.button("🔎 Buscar respuesta", type="primary", use_container_width=True)

# ==========================================================
# TOKENIZACIÓN Y STEMMING
# ==========================================================
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    tokens = [t for t in text.split() if len(t) > 1]
    return [stemmer.stem(t) for t in tokens]

# ==========================================================
# PROCESAMIENTO DE LA BÚSQUEDA
# ==========================================================
if buscar:
    if not documents:
        st.error("Debes cargar o ingresar al menos un documento.")
    elif not question.strip():
        st.error("Escribe una pregunta para realizar la búsqueda.")
    else:
        try:
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_and_stem,
                token_pattern=None,
                min_df=1
            )
            X = vectorizer.fit_transform(documents)
            question_vec = vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, X).flatten()

            best_idx = int(similarities.argmax())
            best_doc = documents[best_idx]
            best_score = float(similarities[best_idx])

            st.markdown(
                '<div class="resultado">'
                '<div class="resultado-titulo">✨ Documento más relacionado</div>',
                unsafe_allow_html=True
            )
            st.write(f"**Tu pregunta:** {question}")
            st.write(f"**Documento seleccionado:** Documento {best_idx + 1}")
            st.info(best_doc)

            metric1, metric2 = st.columns(2)
            with metric1:
                st.metric("📈 Similitud", f"{best_score:.3f}")
            with metric2:
                st.metric("🎯 Relevancia", f"{best_score * 100:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs([
                "📊 Ranking de documentos",
                "🔢 Matriz TF-IDF",
                "✂️ Palabras procesadas"
            ])

            with tab1:
                st.markdown("### Documentos ordenados por relevancia")
                sim_df = pd.DataFrame({
                    "Documento": [f"Documento {i + 1}" for i in range(len(documents))],
                    "Texto": documents,
                    "Similitud": similarities
                }).sort_values("Similitud", ascending=False)
                sim_df["Similitud"] = sim_df["Similitud"].round(3)
                sim_df["Relevancia (%)"] = (sim_df["Similitud"] * 100).round(1)
                st.dataframe(sim_df, use_container_width=True, hide_index=True)

            with tab2:
                st.markdown("### Matriz TF-IDF")
                st.write("Cada fila representa un documento y cada columna una palabra procesada.")
                df_tfidf = pd.DataFrame(
                    X.toarray(),
                    columns=vectorizer.get_feature_names_out(),
                    index=[f"Documento {i + 1}" for i in range(len(documents))]
                )
                st.dataframe(df_tfidf.round(3), use_container_width=True)

            with tab3:
                st.markdown("### Palabras procesadas")
                st.write("El stemming transforma las palabras a una forma reducida para facilitar la comparación.")
                q_stems = tokenize_and_stem(question)
                vocab = vectorizer.get_feature_names_out()
                matched = [
                    stem for stem in q_stems
                    if stem in vocab and df_tfidf.iloc[best_idx].get(stem, 0) > 0
                ]
                st.write("**Palabras procesadas de la pregunta:**")
                st.write(q_stems)
                st.write("**Palabras encontradas en el documento seleccionado:**")
                st.write(matched)

            with st.expander("🧠 ¿Cómo se realizó la búsqueda?"):
                st.markdown("""
                1. **Tokenización:** el texto se divide en palabras.
                2. **Stemming:** las palabras se reducen a una forma básica.
                3. **TF-IDF:** cada documento se convierte en un vector numérico.
                4. **Vectorización:** la pregunta se transforma usando el mismo vocabulario.
                5. **Similitud coseno:** se compara la pregunta con cada documento.
                6. **Ranking:** los documentos se ordenan por similitud.
                7. **Resultado:** se muestra el documento con mayor similitud.
                """)
        except ValueError:
            st.warning("No se encontraron palabras para comparar. Intenta formular la pregunta de otra manera.")

# ==========================================================
# PIE DE PÁGINA
# ==========================================================
st.markdown("""
<div class="footer">
    💙 <strong>EAFITRESPONDE</strong><br>
    Búsqueda inteligente de información mediante procesamiento de lenguaje natural.
</div>
""", unsafe_allow_html=True)
