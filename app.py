import streamlit as st
import os
import json
import hashlib
import re
import io
from pathlib import Path
from typing import List, Dict, Tuple

import requests
import chromadb
import PyPDF2
import docx
import openpyxl
import chardet


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="SimpleRAG",
    page_icon="💬",
    layout="wide"
)

# Ollama Cloud
OLLAMA_CLOUD_URL = "https://ollama.com"

# Modelos Cloud
# Puedes cambiarlos si tienes otro modelo disponible.
DEFAULT_CHAT_MODEL = "gpt-oss:20b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding"

# RAG
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_NUM_CHUNKS = 5
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MEMORY_SIZE = 3

# Límites para evitar que Streamlit Cloud tarde demasiado
MAX_FILE_SIZE_MB = 25
MAX_SUMMARY_CHARS = 12000


# ============================================================
# OLLAMA CLOUD
# ============================================================

def get_ollama_api_key():
    """
    Obtiene la API Key desde Streamlit Secrets.
    Si no existe, intenta obtenerla desde variables de entorno.
    """

    try:
        key = st.secrets.get("OLLAMA_API_KEY")

        if key:
            return key
    except Exception:
        pass

    return os.getenv("OLLAMA_API_KEY")


def get_ollama_headers():
    """
    Headers necesarios para Ollama Cloud.
    """

    api_key = get_ollama_api_key()

    if not api_key:
        return None

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }


def validate_ollama_key():
    """
    Verifica que exista la API Key.
    No realiza llamadas innecesarias a /api/tags.
    """

    key = get_ollama_api_key()

    if not key:
        return False, "No se encontró OLLAMA_API_KEY en Streamlit Secrets."

    return True, "API Key encontrada."


# ============================================================
# OLLAMA CHAT
# ============================================================

def ollama_chat(
    model: str,
    prompt: str,
    temperature: float = 0.1,
    stream: bool = False
):
    """
    Llama directamente a Ollama Cloud.

    Endpoint:
        https://ollama.com/api/chat
    """

    headers = get_ollama_headers()

    if not headers:
        st.error(
            "No se encontró OLLAMA_API_KEY. "
            "Agrega OLLAMA_API_KEY en Streamlit Secrets."
        )
        return None

    url = f"{OLLAMA_CLOUD_URL}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "stream": stream
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=stream,
            timeout=300
        )

        response.raise_for_status()

        return response

    except requests.exceptions.HTTPError as e:

        error_text = ""

        try:
            error_text = response.text[:1000]
        except Exception:
            pass

        st.error(
            f"Error HTTP de Ollama Cloud: {e}\n\n"
            f"{error_text}"
        )

        return None

    except requests.exceptions.RequestException as e:

        st.error(
            f"No se pudo conectar con Ollama Cloud:\n{e}"
        )

        return None


# ============================================================
# EMBEDDINGS
# ============================================================

def embed_texts(
    model: str,
    texts: List[str]
):
    """
    Genera embeddings en una sola llamada usando /api/embed.

    Esto es mucho más eficiente que hacer una petición HTTP
    independiente para cada chunk.
    """

    if not texts:
        return []

    headers = get_ollama_headers()

    if not headers:
        st.error("No existe OLLAMA_API_KEY.")
        return None

    url = f"{OLLAMA_CLOUD_URL}/api/embed"

    payload = {
        "model": model,
        "input": texts
    }

    try:

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        embeddings = data.get("embeddings")

        if not embeddings:

            st.error(
                "Ollama Cloud no devolvió embeddings."
            )

            return None

        return embeddings

    except requests.exceptions.HTTPError as e:

        error_text = ""

        try:
            error_text = response.text[:1000]
        except Exception:
            pass

        st.error(
            f"Error HTTP generando embeddings:\n"
            f"{e}\n\n"
            f"{error_text}"
        )

        return None

    except requests.exceptions.RequestException as e:

        st.error(
            f"Error generando embeddings:\n{e}"
        )

        return None


def embed_text(
    model: str,
    text: str
):
    """
    Embedding de una sola consulta.
    """

    result = embed_texts(
        model,
        [text]
    )

    if not result:
        return None

    return result[0]


# ============================================================
# LECTURA DE ARCHIVOS
# ============================================================

def read_pdf(file):
    text_parts = []

    try:

        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text_parts.append(page_text)

        return "\n".join(text_parts)

    except Exception as e:

        st.error(
            f"Error leyendo PDF {file.name}: {e}"
        )

        return ""


def read_docx(file):
    try:

        document = docx.Document(file)

        paragraphs = [
            p.text
            for p in document.paragraphs
            if p.text.strip()
        ]

        return "\n".join(paragraphs)

    except Exception as e:

        st.error(
            f"Error leyendo DOCX {file.name}: {e}"
        )

        return ""


def read_excel(file):
    try:

        workbook = openpyxl.load_workbook(
            file,
            read_only=True,
            data_only=True
        )

        text_parts = []

        for sheet in workbook.worksheets:

            text_parts.append(
                f"\n--- HOJA: {sheet.title} ---\n"
            )

            for row in sheet.iter_rows(values_only=True):

                values = [
                    str(value)
                    for value in row
                    if value is not None
                ]

                if values:
                    text_parts.append(
                        " | ".join(values)
                    )

        return "\n".join(text_parts)

    except Exception as e:

        st.error(
            f"Error leyendo Excel {file.name}: {e}"
        )

        return ""


def read_text_file(file):

    try:

        raw_data = file.read()

        detected = chardet.detect(raw_data)

        encoding = detected.get(
            "encoding",
            "utf-8"
        )

        return raw_data.decode(
            encoding,
            errors="replace"
        )

    except Exception as e:

        st.error(
            f"Error leyendo {file.name}: {e}"
        )

        return ""


def read_uploaded_file(file):

    extension = Path(file.name).suffix.lower()

    if extension == ".pdf":

        return read_pdf(file)

    elif extension == ".docx":

        return read_docx(file)

    elif extension in [".xlsx", ".xlsm", ".xltx"]:

        return read_excel(file)

    elif extension in [".txt", ".csv", ".md"]:

        return read_text_file(file)

    else:

        return read_text_file(file)


# ============================================================
# LIMPIEZA DEL TEXTO
# ============================================================

def clean_text(text: str):

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int
):

    text = clean_text(text)

    if not text:
        return []

    if chunk_overlap >= chunk_size:

        chunk_overlap = int(
            chunk_size * 0.15
        )

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


# ============================================================
# HASH DE ARCHIVOS
# ============================================================

def get_file_hash(file):

    file.seek(0)

    content = file.read()

    file.seek(0)

    return hashlib.md5(content).hexdigest()


# ============================================================
# RESUMEN DE DOCUMENTOS
# ============================================================

def summarize_document(
    model: str,
    text: str
):

    if not text:
        return ""

    # Evita enviar documentos enormes
    text_for_summary = text[:MAX_SUMMARY_CHARS]

    prompt = f"""
Eres un asistente especializado en análisis documental.

Resume el siguiente documento de forma concisa.

Incluye:
- Tema principal
- Información importante
- Datos relevantes
- Procesos
- Fechas
- Personas o áreas involucradas
- Conclusiones importantes

No inventes información.

DOCUMENTO:

{text_for_summary}
"""

    response = ollama_chat(
        model=model,
        prompt=prompt,
        temperature=0.0,
        stream=False
    )

    if response is None:
        return ""

    try:

        data = response.json()

        message = data.get(
            "message",
            {}
        )

        return message.get(
            "content",
            ""
        ).strip()

    except Exception:

        return ""


# ============================================================
# KEYWORD SEARCH
# ============================================================

def tokenize(text):

    return set(
        re.findall(
            r"\b\w+\b",
            text.lower()
        )
    )


def keyword_score(
    query: str,
    document: str
):

    query_words = tokenize(query)
    document_words = tokenize(document)

    if not query_words:
        return 0.0

    intersection = (
        query_words & document_words
    )

    return len(intersection) / len(
        query_words
    )


# ============================================================
# NORMALIZACIÓN
# ============================================================

def normalize_scores(scores):

    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:

        return [
            1.0 if maximum > 0 else 0.0
            for _ in scores
        ]

    return [
        (score - minimum) /
        (maximum - minimum)
        for score in scores
    ]


# ============================================================
# CHROMADB
# ============================================================

def create_vector_collection():

    client = chromadb.Client()

    collection_name = (
        "simplerag_collection"
    )

    # Si existe, eliminarla
    try:

        client.delete_collection(
            collection_name
        )

    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name
    )

    return collection


# ============================================================
# PROCESAMIENTO RAG
# ============================================================

def process_documents(
    uploaded_files,
    embedding_model,
    chat_model,
    chunk_size,
    chunk_overlap
):

    collection = create_vector_collection()

    all_chunks = []
    all_metadata = []
    all_ids = []

    summaries = {}
    documents_info = []

    progress = st.progress(0)

    total_files = len(
        uploaded_files
    )

    for file_index, file in enumerate(
        uploaded_files
    ):

        # ----------------------------------------
        # Tamaño
        # ----------------------------------------

        file.seek(0)

        file_bytes = file.read()

        file.seek(0)

        file_size_mb = (
            len(file_bytes) /
            1024 /
            1024
        )

        if file_size_mb > MAX_FILE_SIZE_MB:

            st.warning(
                f"{file.name} supera "
                f"{MAX_FILE_SIZE_MB} MB y fue omitido."
            )

            continue

        # ----------------------------------------
        # Hash
        # ----------------------------------------

        file_hash = get_file_hash(
            file
        )

        # ----------------------------------------
        # Lectura
        # ----------------------------------------

        with st.spinner(
            f"Leyendo {file.name}..."
        ):

            text = read_uploaded_file(
                file
            )

        text = clean_text(text)

        if not text:

            st.warning(
                f"No se encontró texto en {file.name}."
            )

            continue

        # ----------------------------------------
        # Resumen
        # ----------------------------------------

        with st.spinner(
            f"Generando resumen de {file.name}..."
        ):

            summary = summarize_document(
                chat_model,
                text
            )

        summaries[file.name] = summary

        # ----------------------------------------
        # Chunks
        # ----------------------------------------

        chunks = create_chunks(
            text,
            chunk_size,
            chunk_overlap
        )

        documents_info.append({
            "name": file.name,
            "hash": file_hash,
            "characters": len(text),
            "chunks": len(chunks)
        })

        # ----------------------------------------
        # Guardar chunks
        # ----------------------------------------

        for chunk_index, chunk in enumerate(
            chunks
        ):

            chunk_id = (
                f"{file_hash}_"
                f"{chunk_index}"
            )

            all_chunks.append(chunk)

            all_metadata.append({
                "source": file.name,
                "chunk": chunk_index
            })

            all_ids.append(
                chunk_id
            )

        progress.progress(
            (file_index + 1) /
            total_files
        )

    progress.empty()

    if not all_chunks:

        st.error(
            "No se pudieron procesar documentos."
        )

        return None, {}, []

    # ========================================================
    # EMBEDDINGS
    # ========================================================

    st.info(
        f"Generando embeddings para "
        f"{len(all_chunks)} chunks..."
    )

    # Una sola llamada por bloques.
    # Esto evita cientos de requests individuales.

    embeddings = []

    batch_size = 32

    embedding_progress = st.progress(0)

    total_batches = (
        len(all_chunks) +
        batch_size - 1
    ) // batch_size

    for batch_number, start in enumerate(
        range(
            0,
            len(all_chunks),
            batch_size
        )
    ):

        batch = all_chunks[
            start:start + batch_size
        ]

        batch_embeddings = embed_texts(
            embedding_model,
            batch
        )

        if not batch_embeddings:

            st.error(
                "No se pudieron generar los embeddings."
            )

            return None, {}, []

        embeddings.extend(
            batch_embeddings
        )

        embedding_progress.progress(
            (batch_number + 1) /
            total_batches
        )

    embedding_progress.empty()

    # ========================================================
    # CHROMA
    # ========================================================

    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadata
    )

    return (
        collection,
        summaries,
        documents_info
    )


# ============================================================
# BÚSQUEDA HÍBRIDA
# ============================================================

def hybrid_search(
    collection,
    query,
    embedding_model,
    num_chunks
):

    if collection is None:

        return []

    # ----------------------------------------
    # Query embedding
    # ----------------------------------------

    query_embedding = embed_text(
        embedding_model,
        query
    )

    if not query_embedding:

        return []

    # ----------------------------------------
    # Solicitar solamente candidatos razonables
    # ----------------------------------------

    collection_count = (
        collection.count()
    )

    candidate_count = min(
        max(num_chunks * 4, 10),
        collection_count
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=candidate_count,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = (
        results.get("documents", [[]])[0]
    )

    metadatas = (
        results.get("metadatas", [[]])[0]
    )

    distances = (
        results.get("distances", [[]])[0]
    )

    if not documents:

        return []

    # ----------------------------------------
    # Semantic score
    # ----------------------------------------

    semantic_scores = [
        1 / (1 + distance)
        for distance in distances
    ]

    # ----------------------------------------
    # Keyword score
    # ----------------------------------------

    keyword_scores = [
        keyword_score(
            query,
            document
        )
        for document in documents
    ]

    semantic_normalized = normalize_scores(
        semantic_scores
    )

    keyword_normalized = normalize_scores(
        keyword_scores
    )

    # ----------------------------------------
    # Hybrid score
    # ----------------------------------------

    combined = []

    for i in range(
        len(documents)
    ):

        score = (
            0.70 *
            semantic_normalized[i]
            +
            0.30 *
            keyword_normalized[i]
        )

        combined.append({
            "document": documents[i],
            "metadata": metadatas[i],
            "semantic_score": semantic_normalized[i],
            "keyword_score": keyword_normalized[i],
            "score": score
        })

    combined.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return combined[
        :num_chunks
    ]


# ============================================================
# PROMPT RAG
# ============================================================

def build_rag_prompt(
    question,
    retrieved_chunks,
    summaries,
    chat_history,
    prompt_template
):

    context_parts = []

    for index, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        source = item[
            "metadata"
        ].get(
            "source",
            "Documento"
        )

        context_parts.append(
            f"""
--- FUENTE {index}: {source} ---

{item["document"]}
"""
        )

    context = "\n".join(
        context_parts
    )

    summary_text = ""

    for filename, summary in summaries.items():

        if summary:

            summary_text += (
                f"\n--- {filename} ---\n"
                f"{summary}\n"
            )

    history_text = ""

    for message in chat_history[-6:]:

        role = message.get(
            "role",
            ""
        )

        content = message.get(
            "content",
            ""
        )

        history_text += (
            f"{role.upper()}: "
            f"{content}\n"
        )

    if not prompt_template.strip():

        prompt_template = """
Responde la pregunta utilizando exclusivamente
la información encontrada en los documentos.

Si la información no está disponible,
indica claramente que no se encuentra en
los documentos proporcionados.

No inventes datos.

Responde en español de manera clara,
profesional y concisa.
"""

    prompt = f"""
{prompt_template}

========================
CONTEXTO RECUPERADO
========================

{context}

========================
RESÚMENES
========================

{summary_text}

========================
HISTORIAL
========================

{history_text}

========================
PREGUNTA
========================

{question}

========================
RESPUESTA
========================
"""

    return prompt


# ============================================================
# PARSE STREAM
# ============================================================

def stream_ollama_response(
    response
):

    for line in response.iter_lines():

        if not line:
            continue

        try:

            decoded = line.decode(
                "utf-8"
            )

            data = json.loads(
                decoded
            )

            message = data.get(
                "message",
                {}
            )

            content = message.get(
                "content",
                ""
            )

            if content:
                yield content

        except Exception:

            continue


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():

    defaults = {

        "messages": [],

        "collection": None,

        "summaries": {},

        "documents_info": [],

        "data_processed": False,

        "embedding_model":
            DEFAULT_EMBEDDING_MODEL,

        "chat_model":
            DEFAULT_CHAT_MODEL,

    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[
                key
            ] = value


initialize_session_state()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuración")

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    st.subheader(
        "☁️ Ollama Cloud"
    )

    key_ok, key_message = (
        validate_ollama_key()
    )

    if key_ok:

        st.success(
            "OLLAMA_API_KEY detectada"
        )

    else:

        st.error(
            key_message
        )

    st.caption(
        "La aplicación utiliza Ollama Cloud. "
        "No necesita Ollama instalado en tu PC."
    )

    # --------------------------------------------------------
    # MODELOS
    # --------------------------------------------------------

    st.subheader(
        "🤖 Modelos"
    )

    chat_model = st.text_input(
        "Modelo de Chat",
        value=st.session_state.chat_model,
        help=(
            "Modelo utilizado para responder "
            "las preguntas y crear resúmenes."
        )
    )

    embedding_model = st.text_input(
        "Modelo de Embeddings",
        value=st.session_state.embedding_model,
        help=(
            "Modelo utilizado para convertir "
            "los documentos en vectores."
        )
    )

    st.session_state.chat_model = (
        chat_model
    )

    st.session_state.embedding_model = (
        embedding_model
    )

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    st.subheader(
        "📚 Configuración RAG"
    )

    chunk_size = st.slider(
        "Tamaño del chunk",
        min_value=500,
        max_value=2000,
        value=DEFAULT_CHUNK_SIZE,
        step=100
    )

    chunk_overlap = st.slider(
        "Overlap",
        min_value=50,
        max_value=400,
        value=DEFAULT_CHUNK_OVERLAP,
        step=50
    )

    num_chunks = st.slider(
        "Chunks recuperados",
        min_value=2,
        max_value=10,
        value=DEFAULT_NUM_CHUNKS
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_TEMPERATURE,
        step=0.05
    )

    memory_size = st.slider(
        "Memoria de conversación",
        min_value=1,
        max_value=10,
        value=DEFAULT_MEMORY_SIZE
    )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    st.subheader(
        "📝 Prompt"
    )

    prompt_template = st.text_area(
        "Instrucciones del asistente",
        value="""
Responde la pregunta utilizando exclusivamente
la información encontrada en los documentos.

Si la información no está disponible,
indica claramente que no se encuentra en
los documentos proporcionados.

No inventes datos.

Responde en español de manera clara,
profesional y concisa.
""",
        height=180
    )

    # --------------------------------------------------------
    # ARCHIVOS
    # --------------------------------------------------------

    st.subheader(
        "📂 Documentos"
    )

    uploaded_files = st.file_uploader(
        "Sube tus documentos",
        type=[
            "pdf",
            "docx",
            "xlsx",
            "xlsm",
            "txt",
            "csv",
            "md"
        ],
        accept_multiple_files=True
    )

    process_button = st.button(
        "🚀 Procesar documentos",
        type="primary",
        use_container_width=True
    )

    if st.button(
        "🗑️ Limpiar conversación",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# PROCESAR DOCUMENTOS
# ============================================================

if process_button:

    if not key_ok:

        st.error(
            "Primero configura OLLAMA_API_KEY "
            "en Streamlit Secrets."
        )

        st.stop()

    if not uploaded_files:

        st.warning(
            "Sube al menos un documento."
        )

        st.stop()

    with st.spinner(
        "Procesando documentos..."
    ):

        (
            collection,
            summaries,
            documents_info
        ) = process_documents(
            uploaded_files=uploaded_files,
            embedding_model=embedding_model,
            chat_model=chat_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    if collection is not None:

        st.session_state.collection = (
            collection
        )

        st.session_state.summaries = (
            summaries
        )

        st.session_state.documents_info = (
            documents_info
        )

        st.session_state.data_processed = (
            True
        )

        st.success(
            "✅ Documentos procesados correctamente."
        )


# ============================================================
# TÍTULO PRINCIPAL
# ============================================================

st.title(
    "💬 SimpleRAG"
)

st.caption(
    "RAG documental con Streamlit + Ollama Cloud + ChromaDB"
)


# ============================================================
# ESTADO DEL SISTEMA
# ============================================================

if st.session_state.data_processed:

    total_chunks = (
        st.session_state.collection.count()
        if st.session_state.collection
        else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Documentos",
            len(
                st.session_state.documents_info
            )
        )

    with col2:

        st.metric(
            "Chunks",
            total_chunks
        )

    with col3:

        st.metric(
            "Modelo",
            chat_model
        )


# ============================================================
# DOCUMENTOS PROCESADOS
# ============================================================

if st.session_state.documents_info:

    with st.expander(
        "📚 Ver documentos procesados"
    ):

        for document in (
            st.session_state.documents_info
        ):

            st.write(
                f"**{document['name']}**"
            )

            st.caption(
                f"Caracteres: "
                f"{document['characters']:,} | "
                f"Chunks: "
                f"{document['chunks']}"
            )


# ============================================================
# CHAT
# ============================================================

if not st.session_state.data_processed:

    st.info(
        "👈 Sube uno o varios documentos "
        "desde el panel lateral y presiona "
        "**Procesar documentos**."
    )

else:

    st.subheader(
        "💬 Pregunta sobre tus documentos"
    )

    # Mostrar historial
    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Escribe tu pregunta..."
    )

    if question:

        # --------------------------------------------
        # Usuario
        # --------------------------------------------

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        # --------------------------------------------
        # Retrieval
        # --------------------------------------------

        with st.spinner(
            "🔎 Buscando información relevante..."
        ):

            retrieved_chunks = (
                hybrid_search(
                    collection=(
                        st.session_state.collection
                    ),
                    query=question,
                    embedding_model=embedding_model,
                    num_chunks=num_chunks
                )
            )

        if not retrieved_chunks:

            answer = (
                "No encontré información "
                "relevante en los documentos."
            )

            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    answer
                )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            st.stop()

        # --------------------------------------------
        # Prompt
        # --------------------------------------------

        prompt = build_rag_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks,
            summaries=(
                st.session_state.summaries
            ),
            chat_history=(
                st.session_state.messages[
                    -(memory_size * 2):
                ]
            ),
            prompt_template=prompt_template
        )

        # --------------------------------------------
        # Chat Cloud
        # --------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            response = ollama_chat(
                model=chat_model,
                prompt=prompt,
                temperature=temperature,
                stream=True
            )

            if response is None:

                st.stop()

            answer_placeholder = st.empty()

            answer = ""

            for token in stream_ollama_response(
                response
            ):

                answer += token

                answer_placeholder.markdown(
                    answer
                )

        # --------------------------------------------
        # Guardar respuesta
        # --------------------------------------------

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        # --------------------------------------------
        # Fuentes
        # --------------------------------------------

        with st.expander(
            "🔎 Ver información recuperada"
        ):

            for index, item in enumerate(
                retrieved_chunks,
                start=1
            ):

                source = item[
                    "metadata"
                ].get(
                    "source",
                    "Documento"
                )

                score = item[
                    "score"
                ]

                st.markdown(
                    f"### {index}. {source}"
                )

                st.caption(
                    f"Relevancia: "
                    f"{score:.2%}"
                )

                st.write(
                    item["document"]
                )

                st.divider()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SimpleRAG • Streamlit Cloud • Ollama Cloud • ChromaDB"
)
