# import os
# import tempfile
# from threading import Thread

# import streamlit as st
# import torch
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
#     TextIteratorStreamer,
# )
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# MODEL_NAME = "google/gemma-3-1b-it"
# EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# SYSTEM_PROMPT = (
#     "You are a helpful assistant. Answer the question using only the given "
#     "context. If the answer is not in the context, say you could not find it."
# )

# st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="centered")


# # ---------- Cached heavy resources (loaded once, not on every rerun) ----------
# @st.cache_resource(show_spinner="Loading embedding model...")
# def load_embeddings():
#     return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# @st.cache_resource(show_spinner="Loading language model (first time takes a while)...")
# def load_llm():
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         llm_int8_enable_fp32_cpu_offload=True,
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_NAME,
#         quantization_config=bnb_config,
#         device_map="auto",
#         max_memory={0: "5GiB", "cpu": "14GiB"},
#         low_cpu_mem_usage=True,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     return model, tokenizer


# @st.cache_resource(show_spinner="Indexing PDF...")
# def build_retriever(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int, k: int):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(pdf_bytes)
#         tmp_path = tmp.name
#     try:
#         docs = PyPDFLoader(tmp_path).load()
#     finally:
#         os.remove(tmp_path)

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size, chunk_overlap=chunk_overlap
#     )
#     chunks = splitter.split_documents(docs)
#     db = FAISS.from_documents(chunks, load_embeddings())
#     return db.as_retriever(search_type="similarity", search_kwargs={"k": k})


# # ---------- Generation ----------
# def stream_answer(model, tokenizer, context, question, max_new_tokens, temperature):
#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {
#             "role": "user",
#             "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
#         },
#     ]
#     prompt = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     streamer = TextIteratorStreamer(
#         tokenizer, skip_prompt=True, skip_special_tokens=True
#     )
#     thread = Thread(
#         target=model.generate,
#         kwargs=dict(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             do_sample=True,
#             pad_token_id=tokenizer.eos_token_id,
#             streamer=streamer,
#         ),
#     )
#     thread.start()
#     for token_text in streamer:
#         yield token_text
#     thread.join()


# # ---------- Sidebar ----------
# with st.sidebar:
#     st.header("Settings")
#     uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
#     k = st.slider("Chunks to retrieve (k)", 1, 10, 5)
#     chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
#     chunk_overlap = st.slider("Chunk overlap", 0, 400, 180, step=20)
#     max_new_tokens = st.slider("Max new tokens", 100, 1000, 600, step=50)
#     temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.1)
#     if st.button("Clear chat"):
#         st.session_state.messages = []
#         st.rerun()

# # ---------- Main ----------
# st.title("📄 Chat with your PDF")
# st.caption(f"RAG with {MODEL_NAME} + FAISS")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# if uploaded is None:
#     st.info("Upload a PDF from the sidebar to get started.")
#     st.stop()

# retriever = build_retriever(uploaded.getvalue(), chunk_size, chunk_overlap, k)
# model, tokenizer = load_llm()

# # Replay chat history
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])
#         if msg.get("sources"):
#             with st.expander("Sources"):
#                 for s in msg["sources"]:
#                     st.markdown(f"**Page {s['page']}**")
#                     st.caption(s["text"])

# # New question
# if question := st.chat_input("Ask something about the document..."):
#     st.session_state.messages.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(question)

#     with st.chat_message("assistant"):
#         docs = retriever.invoke(question)
#         context = "\n\n".join(d.page_content for d in docs)

#         answer = st.write_stream(
#             stream_answer(
#                 model, tokenizer, context, question, max_new_tokens, temperature
#             )
#         )

#         sources = [
#             {"page": d.metadata.get("page", 0) + 1, "text": d.page_content[:300] + "..."}
#             for d in docs
#         ]
#         with st.expander("Sources"):
#             for s in sources:
#                 st.markdown(f"**Page {s['page']}**")
#                 st.caption(s["text"])

#     st.session_state.messages.append(
#         {"role": "assistant", "content": answer, "sources": sources}
#     )




import os
import tempfile
from threading import Thread

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------- Config (CPU only: no CUDA, no bitsandbytes) ----------
MODEL_NAME = "google/gemma-3-1b-it"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_PDF = r"C:\Users\MY PC\Pictures\documents\Autonomous Revenue Platform Master PRDBRD.pdf"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a document. "
    "Use only the provided context. If the answer is not in the context, say "
    "'I could not find that in the document.' Be clear and concise."
)

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="centered")


# ---------- Cached resources ----------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )


@st.cache_resource(show_spinner="Loading Gemma on CPU (first run downloads the model)...")
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,   # float32 is the safest/fastest dtype on most CPUs
        low_cpu_mem_usage=True,
    )
    model.to("cpu")
    model.eval()
    return model, tokenizer


@st.cache_resource(show_spinner="Reading and indexing PDF...")
def build_retriever(pdf_path: str, chunk_size: int, chunk_overlap: int, k: int, _version: str):
    # _version is part of the cache key so a changed file re-indexes
    docs = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, load_embeddings())
    return db.as_retriever(search_type="similarity", search_kwargs={"k": k}), len(docs), len(chunks)


def save_upload(uploaded_file) -> str:
    """Write the uploaded PDF to a temp file and return its path."""
    tmp_dir = tempfile.gettempdir()
    path = os.path.join(tmp_dir, f"rag_{uploaded_file.file_id}_{uploaded_file.name}")
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
    return path


# ---------- Streaming generation ----------
def stream_answer(model, tokenizer, context, question, max_new_tokens, temperature):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cpu")

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    thread.start()
    for piece in streamer:
        yield piece
    thread.join()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    path_input = st.text_input("...or local PDF path", value=DEFAULT_PDF)
    st.divider()
    k = st.slider("Chunks to retrieve (k)", 1, 10, 4)
    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 180, step=20)
    max_new_tokens = st.slider("Max new tokens", 50, 800, 300, step=50)
    temperature = st.slider("Temperature (0 = deterministic)", 0.0, 1.5, 0.3, step=0.1)
    st.caption(f"Running on CPU · {torch.get_num_threads()} threads")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Main ----------
st.title("📄 Chat with your PDF")
st.caption(f"RAG · {MODEL_NAME} · FAISS · CPU only")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Resolve which PDF to use
pdf_path = None
if uploaded is not None:
    pdf_path = save_upload(uploaded)
elif path_input and os.path.isfile(path_input):
    pdf_path = path_input

if pdf_path is None:
    st.info("Upload a PDF in the sidebar, or enter a valid local PDF path.")
    st.stop()

version = str(os.path.getmtime(pdf_path))
retriever, n_pages, n_chunks = build_retriever(
    pdf_path, chunk_size, chunk_overlap, k, version
)
model, tokenizer = load_llm()

st.success(f"Loaded **{os.path.basename(pdf_path)}**: {n_pages} pages, {n_chunks} chunks")

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**Page {s['page']}**")
                    st.caption(s["text"])

# New question
if question := st.chat_input("Ask something about the document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        answer = st.write_stream(
            stream_answer(
                model, tokenizer, context, question, max_new_tokens, temperature
            )
        )

        sources = [
            {
                "page": d.metadata.get("page", 0) + 1,
                "text": d.page_content[:300].replace("\n", " ") + "...",
            }
            for d in docs
        ]
        with st.expander("📚 Sources"):
            for s in sources:
                st.markdown(f"**Page {s['page']}**")
                st.caption(s["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )