# 🚀 Chatbot History – Generative AI to Agentic AI with LangChain

This repository is a hands-on collection of projects exploring the evolution from **traditional Generative AI pipelines** to **modern Agentic AI systems** using LangChain.

It includes practical implementations of:

- 🔹 QA Chatbots  
- 🔹 Retrieval-Augmented Generation (RAG)  
- 🔹 AI Agents  
- 🔹 Offline LLM deployment (Ollama)  
- 🔹 YouTube & Text Summarization  
- 🔹 Vector Databases (AstraDB)  
- 🔹 LangSmith Tracing & Monitoring  

---

# 📂 Project Structure

```
chatbot_history/
│
├── RAG_QA/                  # Retrieval-Augmented QA system
├── SQL_agent/               # SQL query agent using LLM
├── Search_engine_agent/     # Web search-based AI agent
├── Text_Summerization/      # Long text summarization system
├── Youtube_summerization/   # YouTube transcript summarizer
├── long_doc_rag/            # Large document RAG with AstraDB
├── ollama_chatbot/          # Offline chatbot using Ollama
├── app.py                   # Streamlit app entry
├── requirements.txt
└── README.md
```

---

# 🧠 Projects Overview

## 1️⃣ QA Chatbot (Groq + LangChain)

- Built a Question-Answer chatbot using Groq LLMs
- Implemented prompt engineering
- Migrated from traditional chains to LCEL
- Adjustable temperature & tokens via UI

---

## 2️⃣ LCEL (LangChain Expression Language)

Implemented modern pipeline structure:

```python
chain = prompt | llm | output_parser
```

Benefits:
- Cleaner chaining
- Better modularity
- Production-ready design
- Easier debugging

---

## 3️⃣ Offline Chatbot (Ollama)

- Runs fully offline
- Supports models like Mistral / Llama3
- Streamlit UI
- Configurable temperature & token settings

Ideal for local deployment without cloud APIs.

---

## 4️⃣ RAG System (AstraDB)

Built a scalable Retrieval-Augmented Generation system:

- Document loading
- Text chunking
- HuggingFace embeddings
- AstraDB vector storage
- Semantic retrieval
- Context-aware LLM answers

Handles large documentation efficiently.

---

## 5️⃣ SQL Agent

- Natural language → SQL query generation
- Executes queries
- Returns structured results

Demonstrates tool-based agent execution.

---

## 6️⃣ Search Engine Agent

- Agent with web search tool integration
- Retrieves live information
- Produces grounded answers

Moves beyond static prompt-based systems.

---

## 7️⃣ YouTube & Text Summarization

- Extracts transcripts
- Splits long content into chunks
- Summarizes large videos/texts
- Handles token limitations effectively

---

## 🔎 Observability with LangSmith

Integrated LangSmith for:

- Prompt tracing
- Token usage monitoring
- Latency analysis
- Debugging chain execution

Production-level visibility for LLM applications.

---

# 🛠 Tech Stack

- Python
- LangChain (LCEL)
- Groq API
- Ollama
- AstraDB
- HuggingFace Embeddings
- Streamlit
- LangSmith

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/bharatsaini1/chatbot_history.git
cd chatbot_history
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Create .env File

```
GROQ_API_KEY=your_key
HF_TOKEN=your_token
ASTRA_DB_APPLICATION_TOKEN=your_token
ASTRA_DB_ID=your_id
LANGCHAIN_API_KEY=your_key
```

---

# ▶️ Running the Applications

### Run Streamlit Chatbot

```bash
streamlit run app.py
```

### Run Specific Modules

Navigate to folder and run:

```bash
python main.py
```

---

# 📌 Key Learnings

- RAG significantly improves answer grounding
- LCEL simplifies LangChain architecture
- Agents enable multi-step reasoning
- Offline LLM deployment is practical
- Observability is essential for production AI

---

# 🎯 Future Improvements

- Multi-agent collaboration
- Memory-enhanced agents
- FastAPI backend deployment
- Docker containerization
- Streaming responses

---

# ⭐ Support

If you found this project helpful, consider giving it a star ⭐

---

Built while exploring the transition from  
**Generative AI → Retrieval Systems → Agentic AI**