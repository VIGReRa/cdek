from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import uuid
from typing import Dict

from langchain_community.llms import HuggingFacePipeline
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage

from app.models import ChatRequest, ChatResponse
from app.retriever import load_documents, create_vector_store, get_retriever
from app.agent import create_agent_graph
from app.llm_config import get_embeddings

# Хранилище сессий для поддержания контекста диалога
session_store: Dict[str, list] = {}

def get_llm() -> BaseLanguageModel:
    """
    Возвращает LLM модель.

    По умолчанию используется локальная модель через HuggingFace.
    Для использования других моделей (OpenAI, Anthropic и т.д.)
    измените эту функцию согласно документации LangChain.

    Варианты подключения:

    1. OpenAI:
       from langchain_openai import ChatOpenAI
       return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    2. Локальная модель (Ollama):
       from langchain_community.llms import Ollama
       return Ollama(model="llama2")

    3. Локальная модель (HuggingFace):
       Используется по умолчанию
    """

    # Проверка наличия API ключа OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")

    if openai_key:
        # Используем OpenAI если доступен ключ
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_key)
        except ImportError:
            pass

    # По умолчанию используем локальную модель
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1
    )

    return HuggingFacePipeline(pipeline=pipe)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация приложения при запуске."""
    # Загрузка документов и создание векторного хранилища
    print("Загрузка документов...")
    documents = load_documents()

    print("Создание векторного хранилища...")
    vectorstore = create_vector_store(documents)

    print("Инициализация retriever...")
    retriever = get_retriever(vectorstore)

    print("Загрузка LLM модели...")
    llm = get_llm()

    print("Создание агента...")
    agent = create_agent_graph(llm, retriever)

    # Сохраняем в state приложения
    app.state.retriever = retriever
    app.state.agent = agent
    app.state.llm = llm

    yield

    # Очистка при завершении
    session_store.clear()

app = FastAPI(
    title="CDEK Internship Chatbot",
    description="RAG-агент для консультирования по правилам международной стажировки",
    version="1.0.0",
    lifespan=lifespan
)

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Основной эндпоинт для общения с ботом.

    Поддерживает контекст диалога через session_id.
    Если session_id не передан - создается новая сессия.
    """
    # Генерируем или используем существующий session_id
    session_id = request.session_id or str(uuid.uuid4())

    # Получаем историю сообщений сессии
    if session_id not in session_store:
        session_store[session_id] = []

    history = session_store[session_id]

    # Добавляем новое сообщение пользователя
    history.append(HumanMessage(content=request.message))

    try:
        # Запускаем агент
        agent = app.state.agent

        initial_state = {
            "messages": history[-5:],  # Передаем последние 5 сообщений для контекста
            "context": "",
            "sources": [],
            "needs_clarification": False
        }

        result = agent.invoke(initial_state)

        # Получаем ответ агента
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            response_text = last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            response_text = "Извините, я не могу ответить на этот вопрос."

        # Обновляем историю сессии
        session_store[session_id].append(AIMessage(content=response_text))

        # Ограничиваем размер истории (последние 10 сообщений)
        if len(session_store[session_id]) > 10:
            session_store[session_id] = session_store[session_id][-10:]

        return ChatResponse(
            response=response_text,
            session_id=session_id,
            sources=result.get("sources", [])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса."""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)