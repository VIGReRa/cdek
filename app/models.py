from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    """Запрос к чат-боту."""
    message: str
    session_id: Optional[str] = None  # Для поддержания контекста диалога

class ChatResponse(BaseModel):
    """Ответ чат-бота."""
    response: str
    session_id: str
    sources: List[str] = []  # Источники информации для ответа