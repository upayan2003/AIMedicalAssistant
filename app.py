import streamlit as st
import os
import pandas as pd

# --- IMPORTS ---
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- PAGE CONFIGURATION (Must be first!) ---
st.set_page_config(
    page_title="MedBot AI",
    page_icon="🩺",
    layout="wide"
)

# --- CONFIGURATION ---
API_KEY = st.secrets["gcp"]["groq_api"]

# with open("GroqAPI_Key.txt", "r") as f:
#     API_KEY = f.read().strip()

os.environ["GROQ_API_KEY"] = API_KEY
PERSIST_DIR = "./chroma_db_groq"

@st.cache_resource
def load_medical_knowledge():
    # We use HuggingFace for Embeddings (Runs locally on your CPU = FREE)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        # 1. Load Data
        if not os.path.exists("medical_data.csv"):
            st.error("Data file not found. Please ensure 'medical_data.csv' is in the directory.")
            return None

        df = pd.read_csv("medical_data.csv")
        df["combined_text"] = "Patient: " + df["Patient"].astype(str) + "\nDoctor: " + df["Doctor"].astype(str)
        
        loader = DataFrameLoader(df, page_content_column="combined_text")
        documents = loader.load()

        # 2. Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        # 3. Embed & Store
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=embeddings, # Using free local embeddings
            persist_directory=PERSIST_DIR
        )

    return vectorstore.as_retriever(search_kwargs={"k": 3})

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚠️ Disclaimer")
    st.info(
        "This AI provides information based on past cases. "
        "It is **NOT** a substitute for professional medical advice. "
        "Always consult a doctor for serious conditions."
    )
    
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CONTENT ---
st.title("🩺 MedBot AI")
st.subheader("An intelligent medical assistant powered by Llama 3.3 and RAG")
st.divider()
st.write("Describe your symptoms below to find similar past cases and advice.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    # Set avatar based on role
    avatar = "🧑‍💼" if message["role"] == "user" else "🩺"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Accept user input
if user_input := st.chat_input("Type your symptoms here (e.g., 'severe headache with nausea')..."):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message in chat message container
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_input)

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="🩺"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Consulting medical database..."):
            try:
                retriever = load_medical_knowledge()
                
                if retriever:
                    # --- GROQ LLM SETUP ---
                    llm = ChatGroq(
                        model="llama-3.3-70b-versatile", 
                        temperature=0.3
                    )

                    system_prompt = (
                        "You are an AI Medical Assistant using a database of real cases. "
                        "Use the context to answer the user's question. "
                        "If the context doesn't contain relevant info, say 'I don't have enough information'. "
                        "Keep your answer professional and empathetic."
                        "\n\n"
                        "{context}"
                    )

                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])

                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                    response = rag_chain.invoke({"input": user_input})
                    bot_reply = response["answer"]
                else:
                    bot_reply = "System Error: Could not load knowledge base."

            except Exception as e:
                bot_reply = f"An error occurred: {str(e)}"
        
        message_placeholder.markdown(bot_reply)
    
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})