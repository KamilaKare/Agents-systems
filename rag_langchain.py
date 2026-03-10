#!/usr/bin/env python
"""
rag_langchain_chat_history.py
─────────────────────────────
Retrieval-Augmented Generation demo built *entirely* with LangChain
components + OpenAI.

Key ingredients
---------------
▪️   TextLoader + PyPDFLoader            → read .txt & .pdf  
▪️   RecursiveCharacterTextSplitter      → chunk docs  
▪️   OpenAIEmbeddings                    → embed chunks  
▪️   FAISS vector store (LangChain wrap) → similarity search  
▪️   ConversationalRetrievalChain        → chat w/ memory  
▪️   ConversationBufferMemory            → keeps history

Install
-------
pip install \
    langchain-openai \
    langchain-community \
    faiss-cpu \
    pypdf \
    tqdm
"""

from pathlib import Path
import os, textwrap

# ── LangChain pieces ───────────────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# ───────────────────────────────────────────────────────────────────────

DOCS_DIR        = Path(r"c:\Users\KAMILA\Desktop\Formation LLM\Module 3\Data")            # .txt & .pdf live here
EMBED_MODEL     = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
PERSIST_PATH    = "faiss_store"          # folder created on first run

SYSTEM_PROMPT = (
    "You are a precise, concise tutor. "
    "Answer ONLY from the provided context. "
    "If the answer is missing, say “I don't know.”"
)




# ────────────────────────── 0.  API key  ──────────────────────────────
if "OPENAI_API_KEY" not in os.environ:
    raise SystemExit("👉  export OPENAI_API_KEY=... and run again")

# ────────────────────────── 1.  Load docs  ────────────────────────────
def load_documents():
    loaders = []
    for path in DOCS_DIR.rglob("*.txt"):
        loaders.append(TextLoader(str(path)))
    for path in DOCS_DIR.rglob("*.pdf"):
        loaders.append(PyPDFLoader(str(path)))

    if not loaders:
        raise RuntimeError(f"No .txt or .pdf found inside {DOCS_DIR.absolute()}")

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)

docs = load_documents()

# ────────────────────────── 2.  Vector store  ─────────────────────────
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

if Path(PERSIST_PATH).exists():
    vectordb = FAISS.load_local(
        PERSIST_PATH, embeddings, allow_dangerous_deserialization=True
    )
else:
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(PERSIST_PATH)

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

# ────────────────────────── 3.  Memory  ───────────────────────────────
memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True,
    output_key='answer'
)

# ────────────────────────── 4.  Chain  ────────────────────────────────
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
# Define a custom prompt template
# ✔️ Use both "context" and "question" as expected by the chain
custom_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


# ✔️ Tell the chain to use this prompt and that "context" is the document variable
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)


# ────────────────────────── 5.  Chat loop  ────────────────────────────
if __name__ == "__main__":
    print("🔍  Loading documents from", DOCS_DIR)
    print("🔍  Building vector store (this may take a while) ..."
          )
    print("\n📚  Vector store ready. Ask away!  (Ctrl-C to quit)")

    try:
        while True:
            user_q = input("\n💬  You: ")
            result = rag_chain({"question": user_q})

            answer   = result["answer"]
            sources  = result["source_documents"]

            # Show retrieved context for teaching transparency
            print("\n🔍  Retrieved context")
            print("─" * 60)
            for i, doc in enumerate(sources, 1):
                snippet = doc.page_content[:600].replace("\n", " ")
                print(textwrap.indent(textwrap.fill(snippet, 88), f"[Doc {i}] "))
            print("─" * 60)

            # Assistant reply
            print("🤖  Assistant:\n")
            print(textwrap.fill(answer, width=88))

    except KeyboardInterrupt:
        print("\nBye!")
