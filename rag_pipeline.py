import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import *

PROMPT_TEMPLATE = """You are a helpful Pakistani legal assistant.
Use the context below from the Constitution of Pakistan to answer
the question in simple, clear language.
If the answer is not in the context, say:
"I couldn't find this in the Constitution of Pakistan."

Context:
{context}

Question: {question}

Answer:"""


def load_and_chunk_pdf(pdf_path: str) -> list:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(docs)


def get_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(FAISS_INDEX_PATH):
        print("Loading existing index...")
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    print("Building index...")
    chunks = load_and_chunk_pdf(PDF_PATH)
    print(f"Total chunks: {len(chunks)}")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print("Index saved!")
    return vectorstore


def build_qa_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask_question(chain_tuple, question: str) -> dict:
    chain, retriever = chain_tuple
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    source_pages = sorted(list({
        doc.metadata.get("page", "N/A") for doc in docs
    }))
    return {
        "answer": answer,
        "source_pages": source_pages
    }