import os
from typing import List, Dict
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from app.llm_config import get_embeddings

DATA_DIR = "/app/data"

def load_documents() -> List[Document]:
    """Загружает все текстовые файлы из папки data."""
    documents = []
    
    file_mapping = {
        "general_info.txt": "Общая информация",
        "deadlines.txt": "Сроки и дедлайны",
        "benefits.txt": "Преимущества программы",
        "germany_rules.txt": "Правила для Германии",
        "france_rules.txt": "Правила для Франции"
    }
    
    for filename, source_name in file_mapping.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": source_name, "filename": filename}
                    )
                )
    
    return documents

def create_vector_store(documents: List[Document]) -> Chroma:
    """Создает векторное хранилище из документов."""
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = get_embeddings()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="/app/chroma_db"
    )
    
    return vectorstore

def get_retriever(vectorstore: Chroma):
    """Возвращает retriever для поиска релевантных документов."""
    return vectorstore.as_retriever(search_kwargs={"k": 3})