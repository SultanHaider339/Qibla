import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from pathlib import Path
import os
from datetime import datetime

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="اردو اسلامی کتب چیٹ بوٹ",
    page_icon="static/quran.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= CUSTOM CSS (Beautiful Islamic Look) =========================
st.markdown("""
<style>
    .main-header {font-size: 3.5rem; color: #1E4A3D; text-align: center; font-weight: bold; margin-bottom: 0;}
    .sub-header {font-size: 1.4rem; color: #2E8B57; text-align: center; margin-bottom: 30px;}
    .chat-message {padding: 15px; border-radius: 15px; margin: 10px 0;}
    .user-message {background: linear-gradient(90deg, #e6f5ff, #c8e6ff); border-left: 5px solid #2986cc;}
    .bot-message {background: linear-gradient(90deg, #e8f5e8, #d4edda); border-right: 5px solid #2e8b57;}
    .source-box {background:#f8f9fa; padding:12px; border-radius:10px; border:1px solid #2e8b57; font-size:0.9rem; margin-top:10px;}
    .footer {text-align:center; margin-top:50px; color:#555; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# ========================= TITLE =========================
st.markdown("<h1 class='main-header'>اردو اسلامی کتب چیٹ بوٹ</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>صرف آپ کی اپنی کتابوں سے مستند جوابات ─ کوئی ہیلوسینیشن نہیں!</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.image("https://www.svgrepo.com/show/530444/book-open.svg", width=100)
    st.markdown("### کتابیں اپ لوڈ کریں")
    uploaded_files = st.file_uploader(
        "یہاں .txt یا .docx فائلیں ڈالیں",
        type=["txt", "docx"],
        accept_multiple_files=True,
        help="ہر نئی اپ لوڈ پر انڈیکس خودکار اپ ڈیٹ ہو جائے گا"
    )
    
    st.markdown("---")
    st.markdown("#### موجودہ کتابیں")
    books_dir = Path("books")
    if books_dir.exists():
        files = list(books_dir.glob("*.*"))
        if files:
            for f in files:
                st.write(f"Check {f.name}")
        else:
            st.info("ابھی کوئی کتاب نہیں ہے")
    st.markdown("---")
    st.caption("بنایا گیا ❤️ کے ساتھ آپ کے NLP ٹیچر کی طرف سے")

# ========================= SAVE UPLOADED BOOKS =========================
if uploaded_files:
    os.makedirs("books", exist_ok=True)
    for up_file in uploaded_files:
        file_path = Path("books") / up_file.name
        with open(file_path, "wb") as f:
            f.write(up_file.getbuffer())
    st.success(f"{len(uploaded_files)} فائلز کامیابی سے اپ لوڈ ہو گئیں!")

# ========================= INDEXING FUNCTION =========================
@st.cache_resource(show_spinner="کتابیں انڈیکس ہو رہی ہیں... چند سیکنڈ لگیں گے")
def create_vectorstore():
    books_dir = Path("books")
    if not books_dir.exists() or len(list(books_dir.glob("*.*"))) == 0:
        st.warning("کوئی کتاب نہیں ملی۔ براہ کرم پہلے کتابیں اپ لوڈ کریں۔")
        return None

    docs = []
    for file_path in books_dir.glob("*.*"):
        if file_path.suffix.lower() in [".txt", ".docx"]:
            loader = TextLoader(str(file_path), encoding="utf-8)
            docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = create_vectorstore()

# ========================= LLM SETUP =========================
if vectorstore is None:
    st.stop()

from langchain_groq import ChatGroq

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY سیٹ نہیں ہے۔ Streamlit Secrets میں شامل کریں۔")
    st.stop()

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-70b-versatile",
    temperature=0.2,
    max_tokens=1024
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# ========================= CHAT INTERFACE =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "السلام علیکم ورحمۃ اللہ وبرکاتہ\n\nمیں آپ کی اپ لوڈ کردہ اردو کتابوں سے سوالات کے مستند جوابات دوں گا۔\nآپ کوئی بھی دینی، تاریخی یا فقہی سوال پوچھ سکتے ہیں"}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-message user-message'><strong>آپ:</strong> {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message bot-message'><strong>چیٹ بوٹ:</strong> {msg['content']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("یہاں اپنا سوال لکھیں... (مثال: حضور ﷺ کی ولادت کب اور کہاں ہوئی؟)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='chat-message user-message'><strong>آپ:</strong> {prompt}</div>", unsafe_allow_html=True)

    with st.spinner("کتابوں سے تلاش کر رہا ہوں..."):
        result = qa_chain.invoke({"query": prompt})
        answer = result["result"]
        sources = result["source_documents"]

        response = f"{answer}\n\n**ماخذات:**\n"
        for i, doc in enumerate(sources[:3], 1):
            text = doc.page_content.replace("\n", " ").strip()
            response += f"{i}. {text[:280]}...\n"

        st.markdown(f"<div class='chat-message bot-message'><strong>چیٹ بوٹ:</strong><br>{response}</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ========================= FOOTER =========================
st.markdown("---")
st.markdown("""
<div class='footer'>
    بنایا گیا آپ کے NLP ٹیچر کی طرف سے | مکمل مفت & اوپن سورس<br>
    ماڈل: Llama-3.1-70B @ Groq (بلٹ فاسٹ) | ایمبیڈنگ: multilingual-e5-large
</div>
""", unsafe_allow_html=True)
