import os
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="TCS Health Insurance Assistant", page_icon="🩺")
st.title("🩺 TCS India Health Insurance Assistant")
st.caption("Ask any question about enrolment, plans, premiums, claims, etc.")

# Read API key safely
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Add it in Streamlit Cloud Secrets.")
    st.stop()


@st.cache_resource
def load_rag_system():
    # Load PDF
    loader = PDFPlumberLoader("ins.pdf")
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n\n", "\n\n", "\n", "\t", "|", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    # Vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=OPENAI_API_KEY
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful TCS Health Insurance expert.
Answer questions clearly using only the provided context.
If the answer is not in the context, say: "I could not find that in the provided policy document."
If there is a table, present the plan names clearly.
Be professional and precise."""
        ),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    chain = (
        {
            "context": retriever | (lambda docs: "\n\n---\n\n".join([d.page_content for d in docs])),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# Load RAG system
try:
    rag_chain = load_rag_system()
except Exception as e:
    st.error("Failed to initialize the RAG system.")
    st.exception(e)
    st.stop()


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_prompt = st.chat_input("Ask your question about TCS Health Insurance...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = rag_chain.invoke(user_prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("Something went wrong while answering your question.")
                st.exception(e)
