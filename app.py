import streamlit as st
import os
import time
from pathlib import Path
import chromadb
import requests
import json
import PyPDF2
import docx
import openpyxl
import chardet
from typing import List, Dict, Union
import io
import re
import hashlib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SimpleRAG",
    page_icon="💬",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_OLLAMA_URL = "http://localhost:11434"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_NUM_CHUNKS = 4
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MEMORY_SIZE = 3


# ============================================================
# GET OLLAMA URL
# ============================================================

def get_ollama_url():

    # Streamlit Cloud / Secrets
    try:
        if "OLLAMA_URL" in st.secrets:
            return st.secrets["OLLAMA_URL"].rstrip("/")
    except Exception:
        pass

    # Environment variable
    env_url = os.getenv("OLLAMA_URL")

    if env_url:
        return env_url.rstrip("/")

    # Local development
    return DEFAULT_OLLAMA_URL


# ============================================================
# PROMPT
# ============================================================

DEFAULT_PROMPT_TEMPLATE = """Here are the chunks retrieved based on similarity search from user's question.
They might or might not be directly related to the question:

{context}

Document Summaries:

{summaries}

Chat History:

{memory}

User Question:

{question}

Please provide a comprehensive answer to the user's question based on the given context,
document summaries, and chat history.

If the information is not available in the provided context,
please state that you don't have enough information to answer the question.

Your Answer:"""


# ============================================================
# OLLAMA CONNECTION
# ============================================================

@st.cache_data(show_spinner=False)
def check_ollama_connection(ollama_url: str) -> bool:

    try:

        response = requests.get(
            f"{ollama_url}/api/tags",
            timeout=10
        )

        response.raise_for_status()

        return True

    except requests.RequestException:

        return False


# ============================================================
# GET MODELS
# ============================================================

@st.cache_data(show_spinner=False)
def get_ollama_models(ollama_url: str) -> List[str]:

    try:

        response = requests.get(
            f"{ollama_url}/api/tags",
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        return [
            model["name"]
            for model in data.get("models", [])
        ]

    except requests.RequestException as e:

        return []


# ============================================================
# EMBEDDINGS
# ============================================================

def embed_documents(
    ollama_url: str,
    model: str,
    texts: List[str]
):

    embeddings = []

    for text in texts:

        try:

            response = requests.post(
                f"{ollama_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text
                },
                timeout=120
            )

            response.raise_for_status()

            data = response.json()

            embedding = data.get("embedding")

            if not embedding:

                st.error(
                    "Ollama did not return an embedding."
                )

                return None

            embeddings.append(embedding)

        except requests.RequestException as e:

            st.error(
                f"Error getting embedding: {str(e)}"
            )

            return None

    return embeddings


# ============================================================
# CHAT WITH OLLAMA
# ============================================================

def chat_with_documents(
    ollama_url: str,
    model: str,
    prompt: str,
    temperature: float
):

    try:

        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": True
            },
            stream=True,
            timeout=300
        )

        response.raise_for_status()

        return response

    except requests.RequestException as e:

        st.error(
            f"Error generating response: {str(e)}"
        )

        return None


# ============================================================
# FILE ENCODING
# ============================================================

def detect_encoding(file_content: bytes) -> str:

    result = chardet.detect(file_content)

    encoding = result.get("encoding")

    return encoding or "utf-8"


# ============================================================
# READ FILE
# ============================================================

def read_file(
    file: io.BytesIO,
    filename: str
) -> str:

    _, file_extension = os.path.splitext(filename)

    try:

        # PDF
        if file_extension.lower() == ".pdf":

            pdf_reader = PyPDF2.PdfReader(file)

            text_parts = []

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:

                    text_parts.append(page_text)

            return "\n".join(text_parts)


        # DOCX
        elif file_extension.lower() == ".docx":

            document = docx.Document(file)

            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text
            )


        # XLSX / XLS
        elif file_extension.lower() in [".xlsx", ".xls"]:

            workbook = openpyxl.load_workbook(
                file,
                data_only=True
            )

            values = []

            for sheet in workbook.worksheets:

                for row in sheet.iter_rows():

                    for cell in row:

                        if cell.value is not None:

                            values.append(
                                str(cell.value)
                            )

            return "\n".join(values)


        # TXT / other
        else:

            content = file.read()

            encoding = detect_encoding(content)

            return content.decode(
                encoding,
                errors="replace"
            )

    except Exception as e:

        st.error(
            f"Error reading {filename}: {str(e)}"
        )

        return ""


# ============================================================
# CHUNK TEXT
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:

    chunks = []

    if not text:

        return chunks

    if chunk_overlap >= chunk_size:

        chunk_overlap = chunk_size // 5

    start = 0

    step = chunk_size - chunk_overlap

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        start += step

    return chunks


# ============================================================
# SUMMARIZE DOCUMENT
# ============================================================

def summarize_document(
    ollama_url: str,
    model: str,
    text: str
) -> str:

    prompt = f"""
Please provide a concise summary of the following document.

DOCUMENT:

{text}

SUMMARY:
"""

    response = chat_with_documents(
        ollama_url,
        model,
        prompt,
        0.1
    )

    if not response:

        return "Failed to generate summary."


    summary = ""

    try:

        for line in response.iter_lines():

            if line:

                try:

                    json_response = json.loads(
                        line
                    )

                    if "response" in json_response:

                        summary += json_response["response"]

                except json.JSONDecodeError:

                    continue

    except Exception:

        return "Failed to generate summary."

    return summary.strip()


# ============================================================
# CHAT HISTORY
# ============================================================

def get_chat_history(
    messages: List[Dict[str, str]],
    memory_size: int
) -> str:

    history = messages[-memory_size * 2:]

    formatted_history = []

    for msg in history:

        role = (
            "Human"
            if msg["role"] == "user"
            else "AI"
        )

        formatted_history.append(
            f"{role}: {msg['content']}"
        )

    return "\n".join(
        formatted_history
    )


# ============================================================
# KEYWORD SEARCH
# ============================================================

def keyword_search(
    query: str,
    documents: List[Dict]
):

    query_words = set(
        query.lower().split()
    )

    scores = {}

    for i, doc in enumerate(documents):

        doc_words = set(
            doc["content"].lower().split()
        )

        score = (
            len(query_words.intersection(doc_words))
            / len(query_words)
            if query_words
            else 0
        )

        scores[str(i)] = score

    return scores


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    collection,
    documents: List[Dict],
    num_chunks: int
):

    if not documents:

        return []


    # ----------------------------------------
    # Query embedding
    # ----------------------------------------

    question_embedding = embed_documents(
        st.session_state.ollama_url,
        st.session_state.embedding_model,
        [query]
    )

    if not question_embedding:

        return []


    # ----------------------------------------
    # Semantic search
    # ----------------------------------------

    total_docs = len(documents)

    semantic_results = collection.query(
        query_embeddings=question_embedding,
        n_results=total_docs
    )


    semantic_similarities = {}


    ids = semantic_results.get("ids", [[]])[0]

    distances = semantic_results.get(
        "distances",
        [[]]
    )[0]


    for doc_id, distance in zip(
        ids,
        distances
    ):

        similarity = 1 / (1 + distance)

        semantic_similarities[
            str(doc_id)
        ] = similarity


    # ----------------------------------------
    # Keyword search
    # ----------------------------------------

    keyword_scores = keyword_search(
        query,
        documents
    )


    # ----------------------------------------
    # Normalize semantic scores
    # ----------------------------------------

    semantic_values = list(
        semantic_similarities.values()
    )

    if semantic_values:

        max_semantic = max(
            semantic_values
        )

        min_semantic = min(
            semantic_values
        )

        for doc_id in semantic_similarities:

            if max_semantic != min_semantic:

                semantic_similarities[
                    doc_id
                ] = (
                    semantic_similarities[doc_id]
                    - min_semantic
                ) / (
                    max_semantic
                    - min_semantic
                )

            else:

                semantic_similarities[
                    doc_id
                ] = 0


    # ----------------------------------------
    # Normalize keyword scores
    # ----------------------------------------

    keyword_values = list(
        keyword_scores.values()
    )

    if keyword_values:

        max_keyword = max(
            keyword_values
        )

        min_keyword = min(
            keyword_values
        )

        for doc_id in keyword_scores:

            if max_keyword != min_keyword:

                keyword_scores[
                    doc_id
                ] = (
                    keyword_scores[doc_id]
                    - min_keyword
                ) / (
                    max_keyword
                    - min_keyword
                )

            else:

                keyword_scores[
                    doc_id
                ] = 0


    # ----------------------------------------
    # Combine scores
    # ----------------------------------------

    combined_scores = {}

    for doc_id in semantic_similarities:

        combined_scores[doc_id] = (
            semantic_similarities[doc_id]
            +
            keyword_scores.get(
                doc_id,
                0
            )
        ) / 2


    # ----------------------------------------
    # Sort
    # ----------------------------------------

    sorted_docs = sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


    # ----------------------------------------
    # Top chunks
    # ----------------------------------------

    top_docs = []

    for doc_id, score in sorted_docs[:num_chunks]:

        index = int(doc_id)

        if index >= len(documents):

            continue

        doc = documents[index]

        top_docs.append(
            (
                doc["content"],
                doc["metadata"],
                score
            )
        )

    return top_docs


# ============================================================
# COLLECTION NAME
# ============================================================

def sanitize_collection_name(
    name: str
) -> str:

    sanitized = re.sub(
        r"[^\w-]",
        "_",
        name
    )

    sanitized = re.sub(
        r"^[^a-zA-Z0-9]+",
        "",
        sanitized
    )

    sanitized = re.sub(
        r"[^a-zA-Z0-9]+$",
        "",
        sanitized
    )

    sanitized = sanitized[:63]

    while len(sanitized) < 3:

        sanitized += "_"

    return sanitized


# ============================================================
# FILE HASH
# ============================================================

def get_file_hash(file):

    file.seek(0)

    data = file.read()

    file.seek(0)

    return hashlib.md5(
        data
    ).hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():

    # ----------------------------------------
    # SESSION STATE
    # ----------------------------------------

    if "ollama_url" not in st.session_state:

        st.session_state.ollama_url = (
            get_ollama_url()
        )

    if "models" not in st.session_state:

        st.session_state.models = []

    if "connection_status" not in st.session_state:

        st.session_state.connection_status = False

    if "messages" not in st.session_state:

        st.session_state.messages = []

    if "collection" not in st.session_state:

        st.session_state.collection = None

    if "data_processed" not in st.session_state:

        st.session_state.data_processed = False

    if "documents" not in st.session_state:

        st.session_state.documents = []

    if "summaries" not in st.session_state:

        st.session_state.summaries = {}

    if "processed_file_hashes" not in st.session_state:

        st.session_state.processed_file_hashes = {}

    if "embedding_model" not in st.session_state:

        st.session_state.embedding_model = ""

    if "chat_model" not in st.session_state:

        st.session_state.chat_model = ""

    # ----------------------------------------
    # SIDEBAR
    # ----------------------------------------

    with st.sidebar:

        st.title("💬 SimpleRAG")

        st.header("Settings")


        # ------------------------------------
        # MODEL SETTINGS
        # ------------------------------------

        with st.expander(
            "🤖 Model Settings",
            expanded=True
        ):

            ollama_url_input = st.text_input(
                "Ollama Server URL:",
                value=st.session_state.ollama_url,
                help=(
                    "Example: https://your-ollama-server.com"
                )
            )

            if ollama_url_input:

                ollama_url_input = (
                    ollama_url_input.rstrip("/")
                )

            if (
                ollama_url_input
                != st.session_state.ollama_url
            ):

                st.session_state.ollama_url = (
                    ollama_url_input
                )

                st.session_state.connection_status = False

                st.cache_data.clear()


            # --------------------------------
            # TEST CONNECTION
            # --------------------------------

            if not st.session_state.connection_status:

                if check_ollama_connection(
                    st.session_state.ollama_url
                ):

                    st.session_state.connection_status = True

                    st.session_state.models = (
                        get_ollama_models(
                            st.session_state.ollama_url
                        )
                    )

                else:

                    st.session_state.connection_status = False


            if not st.session_state.connection_status:

                st.error(
                    "❌ Cannot connect to Ollama."
                )

                st.info(
                    "Check the Ollama URL and make sure the server is accessible from the Internet."
                )


            elif not st.session_state.models:

                st.error(
                    "No Ollama models available."
                )


            else:

                st.success(
                    "✅ Ollama connected"
                )


                # ----------------------------
                # EMBEDDING MODEL
                # ----------------------------

                selected_embedding_model = st.selectbox(
                    "Select the embedding model",
                    st.session_state.models,
                    key="embedding_model_select"
                )


                if (
                    st.session_state.embedding_model
                    != selected_embedding_model
                ):

                    st.session_state.embedding_model = (
                        selected_embedding_model
                    )

                    st.session_state.data_processed = False

                    st.session_state.collection = None

                    st.session_state.documents = []

                    st.session_state.summaries = {}

                    st.session_state.processed_file_hashes = {}

                    st.warning(
                        "Embedding model changed. Please reprocess your documents."
                    )


                # ----------------------------
                # CHAT MODEL
                # ----------------------------

                selected_chat_model = st.selectbox(
                    "Select the chat model",
                    st.session_state.models,
                    key="chat_model_select"
                )


                st.session_state.chat_model = (
                    selected_chat_model
                )


                # ----------------------------
                # TEMPERATURE
                # ----------------------------

                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=DEFAULT_TEMPERATURE,
                    step=0.1
                )


        # ------------------------------------
        # PROCESS SETTINGS
        # ------------------------------------

        with st.expander(
            "📄 Process Settings"
        ):

            chunk_size = st.number_input(
                "Chunk size",
                min_value=100,
                max_value=2000,
                value=DEFAULT_CHUNK_SIZE
            )

            chunk_overlap = st.number_input(
                "Chunk overlap",
                min_value=0,
                max_value=500,
                value=DEFAULT_CHUNK_OVERLAP
            )

            num_chunks = st.number_input(
                "Number of chunks to retrieve",
                min_value=1,
                max_value=10,
                value=DEFAULT_NUM_CHUNKS
            )


        # ------------------------------------
        # CHAT SETTINGS
        # ------------------------------------

        with st.expander(
            "💬 Chat Settings"
        ):

            memory_size = st.number_input(
                "Number of messages to keep in memory",
                min_value=1,
                max_value=10,
                value=DEFAULT_MEMORY_SIZE
            )

            prompt_template = st.text_area(
                "Prompt template",
                value=DEFAULT_PROMPT_TEMPLATE
            )


        # ------------------------------------
        # FILE UPLOAD
        # ------------------------------------

        uploaded_files = st.file_uploader(
            "Choose files",
            type=[
                "pdf",
                "docx",
                "xlsx",
                "txt",
                "csv"
            ],
            accept_multiple_files=True
        )


        process_button = st.button(
            "🔄 Process Documents",
            use_container_width=True
        )


        # ------------------------------------
        # CLEAR CHAT
        # ------------------------------------

        if st.button(
            "🧹 Clear Chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()


        # ------------------------------------
        # PROCESS DOCUMENTS
        # ------------------------------------

        if process_button:

            if not uploaded_files:

                st.warning(
                    "Please upload at least one document."
                )

            elif not st.session_state.connection_status:

                st.error(
                    "Connect to Ollama before processing documents."
                )

            elif not st.session_state.embedding_model:

                st.error(
                    "Select an embedding model."
                )

            else:

                with st.spinner(
                    "Processing documents..."
                ):

                    start_time = time.time()

                    progress_bar = st.progress(0)

                    status_text = st.empty()

                    documents = []

                    summaries = {}


                    # ----------------------------
                    # READ FILES
                    # ----------------------------

                    for idx, file in enumerate(
                        uploaded_files
                    ):

                        status_text.text(
                            f"Reading {file.name}..."
                        )

                        file_hash = get_file_hash(
                            file
                        )


                        if (
                            file_hash
                            in st.session_state.processed_file_hashes
                        ):

                            cached = (
                                st.session_state
                                .processed_file_hashes[
                                    file_hash
                                ]
                            )

                            content = cached["content"]

                            summary = cached["summary"]


                        else:

                            content = read_file(
                                file,
                                file.name
                            )


                            if content:

                                summary = summarize_document(
                                    st.session_state.ollama_url,
                                    st.session_state.chat_model,
                                    content
                                )

                            else:

                                summary = ""


                            st.session_state.processed_file_hashes[
                                file_hash
                            ] = {
                                "content": content,
                                "summary": summary
                            }


                        if not content:

                            continue


                        summaries[file.name] = summary


                        chunks = chunk_text(
                            content,
                            int(chunk_size),
                            int(chunk_overlap)
                        )


                        for i, chunk in enumerate(chunks):

                            documents.append(
                                {
                                    "content": chunk,
                                    "metadata": {
                                        "source": file.name,
                                        "chunk": i,
                                        "summary": summary
                                    }
                                }
                            )


                        progress = int(
                            ((idx + 1)
                            / len(uploaded_files))
                            * 30
                        )

                        progress_bar.progress(
                            progress
                        )


                    if not documents:

                        st.error(
                            "No readable content was found."
                        )

                        return


                    # ----------------------------
                    # EMBEDDINGS
                    # ----------------------------

                    status_text.text(
                        "Generating embeddings..."
                    )


                    embeddings = embed_documents(
                        st.session_state.ollama_url,
                        st.session_state.embedding_model,
                        [
                            doc["content"]
                            for doc in documents
                        ]
                    )


                    progress_bar.progress(65)


                    if not embeddings:

                        st.error(
                            "Failed to generate embeddings."
                        )

                        return


                    # ----------------------------
                    # CHROMADB
                    # ----------------------------

                    status_text.text(
                        "Initializing ChromaDB..."
                    )


                    try:

                        chroma_client = (
                            chromadb.Client()
                        )

                        progress_bar.progress(75)


                        base_name = (
                            "document_collection_"
                            +
                            st.session_state.embedding_model
                        )


                        collection_name = (
                            sanitize_collection_name(
                                base_name
                            )
                        )


                        # Delete old collection

                        try:

                            chroma_client.delete_collection(
                                name=collection_name
                            )

                        except Exception:

                            pass


                        # Create collection

                        collection = (
                            chroma_client.create_collection(
                                name=collection_name
                            )
                        )


                        # Add vectors

                        status_text.text(
                            "Indexing documents..."
                        )


                        collection.add(
                            embeddings=embeddings,
                            documents=[
                                doc["content"]
                                for doc in documents
                            ],
                            metadatas=[
                                doc["metadata"]
                                for doc in documents
                            ],
                            ids=[
                                str(i)
                                for i in range(
                                    len(documents)
                                )
                            ]
                        )


                        progress_bar.progress(100)


                        # Save state

                        st.session_state.collection = (
                            collection
                        )

                        st.session_state.documents = (
                            documents
                        )

                        st.session_state.summaries = (
                            summaries
                        )

                        st.session_state.data_processed = (
                            True
                        )


                        processing_time = (
                            time.time()
                            - start_time
                        )


                        status_text.success(
                            f"Processing complete in {processing_time:.2f} seconds."
                        )


                        st.success(
                            f"✅ {len(documents)} chunks indexed."
                        )


                    except Exception as e:

                        st.error(
                            f"ChromaDB error: {str(e)}"
                        )

                        st.session_state.data_processed = False


    # ========================================================
    # CHAT
    # ========================================================

    st.header("💬 Chat")


    # ----------------------------------------
    # DISPLAY HISTORY
    # ----------------------------------------

    for chat_message in (
        st.session_state.messages
    ):

        with st.chat_message(
            chat_message["role"]
        ):

            st.markdown(
                chat_message["content"]
            )


    # ----------------------------------------
    # CHAT INPUT
    # ----------------------------------------

    if not st.session_state.data_processed:

        st.info(
            "💡 Connect to Ollama, select models, upload documents and process them first."
        )

        return


    prompt = st.chat_input(
        "Ask a question about the documents"
    )


    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )


        with st.chat_message("user"):

            st.markdown(prompt)


        with st.chat_message("assistant"):

            with st.spinner("Searching documents..."):

                results = hybrid_search(
                    prompt,
                    st.session_state.collection,
                    st.session_state.documents,
                    int(num_chunks)
                )


            if not results:

                st.error(
                    "No relevant documents found."
                )

                return


            # ------------------------------------
            # CONTEXT
            # ------------------------------------

            context = "\n\n".join(
                [
                    doc[0]
                    for doc in results
                ]
            )


            summaries_text = "\n\n".join(
                [
                    f"{file}: {summary}"
                    for file, summary
                    in st.session_state.summaries.items()
                ]
            )


            chat_history = get_chat_history(
                st.session_state.messages,
                int(memory_size)
            )


            full_prompt = (
                prompt_template.format(
                    context=context,
                    summaries=summaries_text,
                    memory=chat_history,
                    question=prompt
                )
            )


            # ------------------------------------
            # GENERATE ANSWER
            # ------------------------------------

            response = chat_with_documents(
                st.session_state.ollama_url,
                st.session_state.chat_model,
                full_prompt,
                temperature
            )


            if not response:

                st.error(
                    "Failed to generate response."
                )

                return


            full_response = ""

            message_placeholder = st.empty()


            try:

                for line in response.iter_lines():

                    if not line:

                        continue


                    try:

                        json_response = json.loads(
                            line
                        )


                        if "response" in json_response:

                            full_response += (
                                json_response["response"]
                            )


                            message_placeholder.markdown(
                                full_response + "▌"
                            )


                    except json.JSONDecodeError:

                        continue


            except Exception as e:

                st.error(
                    f"Error reading response: {str(e)}"
                )


            message_placeholder.markdown(
                full_response
            )


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response
                }
            )


            # ------------------------------------
            # SOURCES
            # ------------------------------------

            with st.expander(
                "📚 View Relevant Document Chunks"
            ):

                for i, (
                    doc,
                    metadata,
                    score
                ) in enumerate(results):

                    st.markdown(
                        f"**Chunk {i + 1}**"
                    )

                    st.text(doc)

                    st.write(
                        f"Source: {metadata['source']}"
                    )

                    st.write(
                        f"Chunk number: {metadata['chunk']}"
                    )

                    st.write(
                        f"Relevance score: {score:.4f}"
                    )

                    st.markdown("---")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
