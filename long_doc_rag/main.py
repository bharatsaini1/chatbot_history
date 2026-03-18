import os
from dotenv import load_dotenv
from pypdf import PdfReader
import cassio

# LangChain 1.x imports
from langchain_community.vectorstores import Cassandra
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

# -------------------------
# 1️⃣ Load Environment Variables
# -------------------------
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = ""
ASTRA_DB_ID = ""
GROQ_API_KEY = ""

# -------------------------
# 2️⃣ Initialize Astra DB
# -------------------------
cassio.init(
    token=ASTRA_DB_APPLICATION_TOKEN,
    database_id=ASTRA_DB_ID,
)

# -------------------------
# 3️⃣ Read PDF
# -------------------------
pdfreader = PdfReader("long_doc_rag/Think-And-Grow-Rich.pdf")

raw_text = ""
for page in pdfreader.pages:
    content = page.extract_text()
    if content:
        raw_text += content

# -------------------------
# 4️⃣ Split Text (Better splitter for big PDFs)
# -------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

texts = text_splitter.split_text(raw_text)

print(f"Total chunks created: {len(texts)}")

# -------------------------
# 5️⃣ Embeddings (HuggingFace)
# -------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------
# 6️⃣ Create Cassandra Vector Store
# -------------------------
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)

# Insert texts
astra_vector_store.add_texts(texts)

print("Texts inserted into Astra DB")

# -------------------------
# 7️⃣ Create Retriever (Use MMR for Big PDFs)
# -------------------------
retriever = astra_vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)

# -------------------------
# 8️⃣ Groq LLM
# -------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant"
)

# -------------------------
# 9️⃣ Retrieval QA Chain (Modern Replacement of VectorStoreIndexWrapper)
# -------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# -------------------------
# 🔟 Interactive Loop
# -------------------------
while True:
    query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()

    if query_text.lower() == "quit":
        break

    if not query_text:
        continue

    print(f"\nQUESTION: {query_text}")

    response = qa.invoke(query_text)

    print(f"\nANSWER: {response}\n")

    print("Top Relevant Chunks:")
    docs = retriever.invoke(query_text)
    for doc in docs[:3]:
        print("•", doc.page_content[:200], "...\n")
