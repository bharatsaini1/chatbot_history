import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv


load_dotenv()
# load environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# create llm model
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-8b-instant")

# crate chat prompt template

prompt_template = ChatPromptTemplate.from_messages([
    ("system", 
     "Answer the question based only on the provided context. "
     "If you don't know the answer, say you don't know. "
     "<context>{context}</context>"
    ),
    ("human", "{input}")
])

# Create vecotor 

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papper") ## Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  ## Document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)  ## Vector store creation




user_prompt = st.text_input("Ask a question about the research papers")  ## User input

if st.button("Document Embeddings"):
    create_vector_embeddings()  ## Create vector embeddings
    st.write("Vector embeddings created successfully!")


import time
if user_prompt:

    if "vectors" not in st.session_state:
        st.warning("Please click 'Document Embeddings' first.")
        st.stop()

    document_chain = create_stuff_documents_chain(llm, prompt_template)

    retriever = st.session_state.vectors.as_retriever(search_type="mmr", search_kwargs={"k": 8})

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({
        "input": user_prompt
    })

    st.write(response["answer"])

    with st.expander("Source Documents"):
        for doc in response["context"]:
            st.write("SOURCE:", doc.metadata.get("source"))
            st.write(doc.page_content[:300])
            st.write("--" * 20)

        
