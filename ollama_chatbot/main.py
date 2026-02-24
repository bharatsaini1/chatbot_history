import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

# LangSmith tracing
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "QA CHATBOT HISTORY"
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

@st.cache_resource
def get_llm(engine: str, temperature: float, max_tokens: int):
    """Cached LLM creation with proper config."""
    return ChatOllama(
        model=engine,
        temperature=temperature,
        num_predict=max_tokens,  # ChatOllama uses num_predict for max tokens
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar config
st.sidebar.title("Configuration")
model_options = ["llama3", "mistral", "phi3"]  # Common Ollama models
engine = st.sidebar.selectbox("Select Model", model_options)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 50, 500, 200)

# Create chain once
llm = get_llm(engine, temperature, max_tokens)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions accurately."),
    MessagesPlaceholder(variable_name="history"),  # Chat history
    ("human", "{question}")
])
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# UI
st.title("🦙 QA Chatbot with History")
st.write("Ask questions and maintain conversation context!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Chat input
if prompt := st.chat_input("What is your question?"):
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("human"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass history + current question
            response = chain.invoke({
                "question": prompt,
                "history": st.session_state.messages[:-1]  # Exclude current for context
            })
            st.markdown(response)
    
    # Add AI response to history
    st.session_state.messages.append(AIMessage(content=response))