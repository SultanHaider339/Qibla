import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from pathlib import Path
import os

# ===================== PAGE SETUP =====================
st.set_page_config(
    page_title="اردو اسلامی کتب چیٹ بوٹ",
    page_icon="📖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ===================== BEAUTIFUL ISLAMIC DESIGN =====================
st.markdown("""
<style>
    .big-title {font-size: 3.8rem; color: #1e40af; text-align: center; font-weight: bold;}
    .subtitle {font-size: 1.6rem; color: #15803d; text-align: center; margin: 20px 0 40px;}
    .chat-box {background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-radius: 20px; padding: 20px; box-shadow: 0 8px 25px rgba(0,0,0,0.1);}
    .source-box {background: #fefce8; padding: 12px; border-radius: 12px; border-right: 4px solid #ca8a04; margin-top: 15px; font-size: 0.92rem;}
    .footer {text-align: center; margin-top: 60px; color: #64748b; font-size: 0.95rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-title'>اردو اسلامی کتب چیٹ بوٹ</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>صرف مستند کتب سے فوری اور درست جوابات<br>کوئی ہیلوسینیشن نہیں • مکمل پرائیویٹ</p>", unsafe_allow_html=True)

# ===================== LOAD BOOKS (COMPLETELY HIDDEN) =====================
@st.cache_resource(show_spinner="کتابیں لوڈ ہو رہی ہیں... چند سیکنڈ")
def load_books():
    books_path = Path("books")
    if not books_path.exists() or len(list(books_path.iterdir())) == 0:
        return None

    docs = []
    for file in books_path.glob("*.txt"):
        loader = TextLoader(str(file), encoding="utf-8")
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_books()

if vectorstore is None:
    st.warning("کوئی کتاب نہیں ملی۔ مالک سے رابطہ کریں۔")
    st.stop()

# ===================== LLM =====================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("API key missing.")
    st.stop()

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-70b-versatile",
    temperature=0.2
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# ===================== CHAT =====================
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "السلام علیکم ورحمۃ اللہ وبرکاتہ\n\nآپ اپنی مرضی کا کوئی بھی دینی، فقہی، سیرت یا تاریخی سوال اردو میں پوچھ سکتے ہیں۔ میں صرف اپنی محفوظ شدہ مستند کتابوں سے جواب دوں گا۔"
    "
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("یہاں اپنا سوال لکھیں..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("کتابوں سے تلاش کر رہا ہوں..."):
            result = qa.invoke({"query": prompt})
            answer = result["result"]

            st.markdown(answer)

            if result["source_documents"]:
                with st.expander("ماخذ دیکھیں"):
                    for i, doc in enumerate(result["source_documents"][:3], 1):
                        st.markdown(f"<div class='source-box'><strong>ماخذ {i}:</strong> {doc.page_content.strip()[:400]}...</div>", 
                                  unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": answer})

# ===================== FOOTER =====================
st.markdown("---")
st.markdown("""
<div class='footer'>
    مکمل پرائیویٹ • کوئی ڈیٹا شیئر نہیں ہوتا • بنایا گیا آپ کے NLP ٹیچر کی طرف سے<br>
    ماڈل: Llama-3.1-70B (Groq) • ایمبیڈنگ: multilingual-e5-large
</div>
""", unsafe_allow_html=True)
