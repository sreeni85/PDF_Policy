import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

st.set_page_config(page_title="TCS Health Insurance Assistant", page_icon="🩺")
st.title("🩺 TCS India Health Insurance Assistant")
st.caption("Ask any question about enrolment, plans, premiums, claims, etc.")

# Load API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_rag_system():
    # Load PDF
    loader = PDFPlumberLoader("ins.pdf")
    documents = loader.load()

    # Advanced chunking for tables
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n\n", "\n\n", "\n", "\t", "|", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Embeddings + Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful TCS Health Insurance expert.
Answer questions clearly using only the provided context.
If there is a table, list the plans (Silver, Gold, Gold Plus, Platinum, Platinum Plus) properly.
Be professional and precise."""),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    chain = (
        {"context": retriever | (lambda docs: "\n\n---\n\n".join([d.page_content for d in docs])),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Load the RAG system
rag_chain = load_rag_system()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question about TCS Health Insurance..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(prompt)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})