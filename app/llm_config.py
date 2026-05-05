from langchain.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """Возвращает модель эмбеддингов для векторизации текста."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )