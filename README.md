# CDEK Internship Chatbot - RAG Agent на LangGraph

Сервис чат-бота для консультирования пользователей по правилам международной стажировки "CdekStart".

## 🎯 Особенности

- **RAG (Retrieval-Augmented Generation)** - поиск ответов в базе знаний
- **Поддержка контекста диалога** - бот помнит историю общения
- **Уточняющие вопросы** - если запрос неоднозначен (например, не указана страна), бот задаст уточняющий вопрос
- **Без галлюцинаций** - отвечает только на основе предоставленных документов
- **Гибкое подключение LLM** - поддержка OpenAI, локальных моделей (HuggingFace, Ollama)

## 📁 Структура проекта

```
.
├── app/
│   ├── __init__.py
│   ├── agent.py          # LangGraph агент с логикой диалога
│   ├── llm_config.py     # Конфигурация эмбеддингов
│   ├── main.py           # FastAPI приложение
│   ├── models.py         # Pydantic модели
│   └── retriever.py      # Загрузка документов и векторный поиск
├── data/
│   ├── general_info.txt  # Общая информация о программе
│   ├── deadlines.txt     # Сроки и дедлайны
│   ├── benefits.txt      # Преимущества программы
│   ├── germany_rules.txt # Правила для Германии
│   └── france_rules.txt  # Правила для Франции
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 Быстрый старт

### Вариант 1: Запуск через Docker Compose (рекомендуется)

```bash
# Запуск сервиса
docker-compose up --build

# Сервис доступен на http://localhost:8000
```

### Вариант 2: Локальный запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск сервера
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 📡 API

### POST /chat

Основной эндпоинт для общения с ботом.

**Request:**
```json
{
  "message": "Какая ставка стипендии в Германии?",
  "session_id": "optional-session-id"
}
```

**Response:**
```json
{
  "response": "Ставка стипендии в Германии составляет 1200 евро в месяц.",
  "session_id": "generated-or-provided-session-id",
  "sources": ["Правила для Германии"]
}
```

### GET /health

Проверка работоспособности сервиса.

**Response:**
```json
{"status": "ok"}
```

## 🔧 Настройка LLM

По умолчанию используется локальная модель **TinyLlama-1.1B-Chat**. Для использования других моделей:

### OpenAI

1. Установите зависимость: `pip install langchain-openai`
2. Передайте API ключ через переменную окружения:
   ```bash
   export OPENAI_API_KEY=your-api-key
   docker-compose up --build
   ```

### Другие модели

Отредактируйте функцию `get_llm()` в файле `app/main.py`:

```python
# Ollama (локально)
from langchain_community.llms import Ollama
return Ollama(model="llama2")

# Anthropic Claude
from langchain_anthropic import ChatAnthropic
return ChatAnthropic(model="claude-3-sonnet-20240229")
```

## 📝 Примеры запросов

### Запрос с уточнением (бот спросит страну)
```json
{"message": "Какая ставка стипендии?"}
```

### Конкретный запрос
```json
{"message": "Какая ставка стипендии во Франции?"}
```

### Запрос в контексте диалога
```json
{"message": "А какой там налог?", "session_id": "previous-session-id"}
```

### Общие вопросы
```json
{"message": "Когда дедлайн подачи документов?"}
```

## ⚙️ Технические детали

- **Векторное хранилище**: ChromaDB
- **Эмбеддинги**: sentence-transformers/all-MiniLM-L6-v2
- **Оркестрация**: LangGraph
- **API框架**: FastAPI
- **Контейнеризация**: Docker + Docker Compose

## 📄 База знаний

Проект включает 5 документов:

| Файл | Описание |
|------|----------|
| general_info.txt | Общая информация о программе стажировки |
| deadlines.txt | Сроки приема заявок и этапы отбора |
| benefits.txt | Преимущества для стажеров |
| germany_rules.txt | Правила для локации Германия (Берлин) |
| france_rules.txt | Правила для локации Франция (Париж) |

## 🔒 Безопасность

- Не храните API ключи в репозитории
- Используйте переменные окружения для конфиденциальных данных
- В production настройте CORS политики

## 📝 Лицензия

Тестовое задание для стажера LLM Engineer