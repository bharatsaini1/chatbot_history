from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
# Langhsmith API key and tracing configuration
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "QA CHATBOT HISTORY"




#  Prompt template for the chatbot
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided context."),
    ("human", "Question: {question}")
])


def generate_response(question, api_key, llm, temperature, max_tokens):

    if not api_key:
        return "Please enter your Groq API key."

    chat_groq = ChatGroq(
        api_key=api_key,
        model=llm,
        temperature=temperature,
        max_tokens=max_tokens
    )

    output_parser = StrOutputParser()

    chain = prompt | chat_groq | output_parser

    answer = chain.invoke({"question": question})
    return answer



# Title of the Streamlit app
st.title("QA Chatbot with History")

#Sidebar for API key and parameters
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Groq API Key",type="password")

## Drop Down to select various Groq models
model_options = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
llm = st.sidebar.selectbox("Select Groq Model", model_options)

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7 )
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main Interface for user input
st.write("Go ahead and ask any question ")
user_input = st.text_input("Your:")

if user_input:
    response = generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)

else :
    st.write("Please enter a question to get started.")


