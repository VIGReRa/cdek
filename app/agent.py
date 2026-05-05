from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
import os

class AgentState(TypedDict):
    """Состояние агента для LangGraph."""
    messages: List[BaseMessage]
    context: str
    sources: List[str]
    needs_clarification: bool

def create_agent_graph(llm: BaseLanguageModel, retriever):
    """Создает граф агента с поддержкой контекста и уточняющих вопросов."""

    # Промпт для определения необходимости уточнения
    clarification_prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты - ассистент программы международной стажировки CdekStart.
Твоя задача - определить, нуждается ли запрос пользователя в уточнении.

Запрос требует уточнения, если:
1. Пользователь спрашивает о правилах, стипендии, визе или рабочих условиях, но не указал страну (Германия или Франция)
2. Запрос неоднозначен и может относиться к разным локациям

Если запрос требует уточнения - ответь вопросом о том, какая страна интересует пользователя.
Если запрос конкретный - передай его дальше для обработки.

Ответь только YES или NO."""),
        ("human", "{question}")
    ])

    # Промпт для генерации ответа
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты - ассистент программы международной стажировки CdekStart.
Отвечай ТОЛЬКО на основе предоставленного контекста из базы знаний.
Если в контексте нет информации для ответа - скажи, что у тебя нет такой информации.
Не выдумывай факты и не используй внешние знания.

Контекст из базы знаний:
{context}

Источники: {sources}"""),
        ("human", "{question}")
    ])

    def check_clarification(state: AgentState) -> AgentState:
        """Проверяет, нужно ли уточнение."""
        last_message = state["messages"][-1].content if state["messages"] else ""

        chain = clarification_prompt | llm
        response = chain.invoke({"question": last_message})
        response_text = str(response.content).strip().upper()

        needs_clarification = "YES" in response_text

        return {
            **state,
            "needs_clarification": needs_clarification
        }

    def ask_clarification(state: AgentState) -> AgentState:
        """Задает уточняющий вопрос о стране."""
        clarifying_message = AIMessage(
            content="Уточните, пожалуйста, какая страна вас интересует: Германия или Франция?"
        )
        return {
            **state,
            "messages": state["messages"] + [clarifying_message],
            "sources": []
        }

    def retrieve_context(state: AgentState) -> AgentState:
        """Извлекает релевантный контекст из базы знаний."""
        last_message = state["messages"][-1].content if state["messages"] else ""

        docs = retriever.invoke(last_message)

        context_parts = []
        sources = []

        for doc in docs:
            context_parts.append(doc.page_content)
            source = doc.metadata.get("source", "Неизвестно")
            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)

        return {
            **state,
            "context": context,
            "sources": sources
        }

    def generate_answer(state: AgentState) -> AgentState:
        """Генерирует ответ на основе контекста."""
        chain = answer_prompt | llm
        response = chain.invoke({
            "context": state["context"],
            "sources": ", ".join(state["sources"]),
            "question": state["messages"][-1].content
        })

        answer_message = AIMessage(content=str(response.content))

        return {
            **state,
            "messages": state["messages"] + [answer_message]
        }

    def should_clarify(state: AgentState) -> str:
        """Определяет, нужно ли задавать уточняющий вопрос."""
        if state["needs_clarification"]:
            return "clarify"
        return "retrieve"

    # Создаем граф
    graph = StateGraph(AgentState)

    # Добавляем узлы
    graph.add_node("check_clarification", check_clarification)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_answer", generate_answer)

    # Добавляем ребра
    graph.set_entry_point("check_clarification")
    graph.add_conditional_edges(
        "check_clarification",
        should_clarify,
        {
            "clarify": "ask_clarification",
            "retrieve": "retrieve_context"
        }
    )
    graph.add_edge("ask_clarification", "END")
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("generate_answer", "END")

    return graph.compile()