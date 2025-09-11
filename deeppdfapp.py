import streamlit as st
import os
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import numpy as np
from typing import List
import asyncio

# Set up event loop for async operations
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="📄 PDF Q&A Assistant", layout="wide", page_icon="🤖")

# Custom CSS
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
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# DEEPSEEK CONFIGURATION
# ---------------------------
DEEPSEEK_API_KEY = "sk-e55bdac1c4ff448aac03abaeb2062140"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ---------------------------
# CUSTOM DEEPSEEK EMBEDDINGS CLASS
# ---------------------------
class DeepSeekEmbeddings(Embeddings):
    def __init__(self, api_key: str, base_url: str = DEEPSEEK_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        self.model = "text-embedding"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents."""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from DeepSeek API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": text,
            "model": self.model
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result['data'][0]['embedding']
        except Exception as e:
            st.error(f"Error getting embedding: {str(e)}")
            return [0.0] * 512  # Return zero vector as fallback

# ---------------------------
# DEEPSEEK CHAT COMPLETION FUNCTION
# ---------------------------
def deepseek_chat_completion(messages: List[dict], temperature: float = 0.1) -> str:
    """Send chat completion request to DeepSeek API."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

# ---------------------------
# CUSTOM DEEPSEEK CHAT MODEL WRAPPER
# ---------------------------
class DeepSeekChatWrapper:
    def __init__(self, api_key: str, base_url: str = DEEPSEEK_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url
        
    def invoke(self, prompt: str) -> str:
        """Invoke the DeepSeek chat model."""
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful AI assistant that answers questions based on the provided context. Be accurate and relevant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        return deepseek_chat_completion(messages)

# ---------------------------
# SESSION STATE
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ---------------------------
# APP TITLE
# ---------------------------
st.title("📄 DeepSeek PDF Q&A Assistant")

# ---------------------------
# FILE UPLOAD
# ---------------------------
file = st.file_uploader("Upload your PDF file:", type="pdf")

if file is not None and st.session_state.vector_store is None:
    with st.spinner("Processing PDF..."):
        file_path = os.path.join("data", file.name)
        os.makedirs("data", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # Initialize DeepSeek embeddings
        embeddings = DeepSeekEmbeddings(api_key=DEEPSEEK_API_KEY)
        
        # Load and process PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len
        )
        final_docs = text_splitter.split_documents(docs)

        # Create vector store
        try:
            vectors = FAISS.from_documents(final_docs, embeddings)
            st.session_state.vector_store = vectors
            st.success(f"✅ File **{file.name}** uploaded and processed successfully!")
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")

# ---------------------------
# CHAT HISTORY DISPLAY
# ---------------------------
for chat in st.session_state.chat_history:
    st.markdown(f"""
        <div class="chat-bubble user-bubble"><b>You:</b> {chat['user']}</div>
        <div class="chat-bubble bot-bubble"><b>Assistant:</b> {chat['bot']}</div>
    """, unsafe_allow_html=True)

# ---------------------------
# CHAT INPUT
# ---------------------------
query = st.text_input("💬 Ask your question:", key="chat_input")
send = st.button("Ask DeepSeek")

if send and query.strip() != "":
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload a PDF file first.")
    else:
        with st.spinner("Thinking..."):
            try:
                # Retrieve relevant documents
                retriever = st.session_state.vector_store.as_retriever()
                relevant_docs = retriever.get_relevant_documents(query)
                
                # Prepare context
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Create prompt
                prompt = f"""
                Answer the following question based on the provided context.
                Be accurate and relevant. If you don't know the answer, say so.

                Context:
                {context}

                Question: {query}

                Answer:
                """
                
                # Get response from DeepSeek
                llm = DeepSeekChatWrapper(api_key=DEEPSEEK_API_KEY)
                answer = llm.invoke(prompt)
                
                # Update chat history
                st.session_state.chat_history.append({"user": query, "bot": answer})
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing your request: {str(e)}")

# ---------------------------
# CLEAR CHAT BUTTON
# ---------------------------
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
