import os 
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import chat_groq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, DuckDuckGoSearchRun,WikipediaQueryRun
from langchain.agents import create_agent, AgentType, Tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler


## Used the inbuilt tool of wikipedia & ARXIV
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search")


# Streamlit app
st.title("Search Engine Agent with Groq API")


# SideBar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant that can answer questions using Wikipedia, ArXiv, and DuckDuckGo search."}
    ]


for message in st.session_state["messages"]:
    st.chat_message(message['role'] ).write(message['content'])


if prompt := st.chat_input(placeholder="What is Machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = chat_groq.ChatGroq(api_key=api_key, model="groq-2b",streaming=True)
    tools = [wiki, arxiv, search]

    search_agent = create_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, callbacks=[StreamlitCallbackHandler()], handling_parsing_errors=True)


    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
        