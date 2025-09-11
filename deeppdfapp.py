import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ---------------------------
# CONFIG
# ---------------------------
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
st.set_page_config(page_title="📄 PDF Q&A Assistant", layout="wide", page_icon="🤖")

# Custom CSS for sticky input and chat style
st.markdown("""
    <style>
        .chat-container {
            display: flex;
            flex-direction: column-reverse;
            max-height: 70vh;
            overflow-y: auto;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #fefefe;
        }
        .chat-bubble {
            padding: 10px;
            margin: 6px 0;
            border-radius: 8px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .user-bubble {
           
            background-color: #dbcdcc;
            color: black;
            align-self: flex-end;
        }
        .bot-bubble {
            
            align-self: flex-start;
        }
        .input-box {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 12px;
            border-top: 2px solid #ddd;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# API KEYS & MODEL
# ---------------------------
Google_api = "AIzaSyDEk75eXsZQRZ2gnkyauHeEW6SOEulnvGk"  # ⚠️ replace with your real key
model = "gemini-2.0-flash"

# ---------------------------
# SESSION STATE
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

# ---------------------------
# APP TITLE
# ---------------------------
st.title("📄 Give answer from your extracted pdf")

# ---------------------------
# FILE UPLOAD
# ---------------------------
file = st.file_uploader("Upload your PDF file:", type="pdf")

if file is not None and st.session_state.retrieval_chain is None:
    file_path = os.path.join("data", file.name)
    os.makedirs("data", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    embeddings = OpenAIEmbeddings(
    model="text-embedding",  # DeepSeek's embedding model
    api_key="sk-e55bdac1c4ff448aac03abaeb2062140",
    base_url="https://api.deepseek.com/v1"  # Add /v1
)
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = text_splitter.split_documents(docs)

    vectors = FAISS.from_documents(final_docs, embeddings)

    llm = ChatOpenAI(
    model="deepseek-chat",  # or try "deepseek-chat:32k" for longer context
    api_key="sk-e55bdac1c4ff448aac03abaeb2062140",
    base_url="https://api.deepseek.com/v1"  # Add /v1
)

    prompt = ChatPromptTemplate.from_template("""
    Answer the given question based on the provided PDF or context.
    Be accurate and relevant.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    st.session_state.retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    st.success(f"✅ File **{file.name}** uploaded and processed successfully!")

# ---------------------------
# CHAT HISTORY DISPLAY
# ---------------------------

for chat in st.session_state.chat_history:  # oldest at top, newest at bottom
    st.markdown(f"""
        <div class="chat-bubble user-bubble"><b>You:</b> {chat['user']}</div>
        <div class="chat-bubble bot-bubble"><b>Assistant:</b> {chat['bot']}</div>
    """, unsafe_allow_html=True)


# ---------------------------
# STICKY INPUT BOX
# ---------------------------
with st.container():
  
    query = st.text_input("💬 Ask your question:", key="chat_input")
    send = st.button("Result")
    if send:
        if file is None:
            st.warning("⚠️ Please upload a PDF file first.")
        elif query.strip() != "":
            action = st.session_state.retrieval_chain.invoke({"input": query})
            answer = action['answer']
            st.session_state.chat_history.append({"user": query, "bot": answer})
            st.rerun()  # refresh to update chat
    



