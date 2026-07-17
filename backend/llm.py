"""
Обёртка над Anthropic API. Ключ берётся из переменной окружения
ANTHROPIC_API_KEY (см. .env.example). Промпт жёстко ограничивает
модель контекстом из найденных чанков — если ответа в документах нет,
бот обязан честно сказать, что не знает, а не выдумывать (без этого
RAG-бот быстро теряет доверие клиентов).
"""
import os
from anthropic import Anthropic

SYSTEM_PROMPT = """Ты — ассистент поддержки компании. Отвечай на вопросы \
клиентов ТОЛЬКО на основе предоставленного контекста из документов компании.

Правила:
1. Если ответа нет в контексте — прямо скажи, что не располагаешь этой \
информацией, и предложи обратиться к менеджеру. Не выдумывай факты.
2. Отвечай кратко и по делу, в дружелюбном тоне.
3. Не упоминай, что ты используешь "контекст" или "документы" — отвечай \
как обычный ассистент компании.
"""


def generate_answer(question: str, context_chunks: list[dict]) -> tuple[str, bool]:
    """
    Возвращает (текст_ответа, answered) — answered=False, если релевантных
    чанков не нашлось (используется для аналитики "бот не смог ответить").
    """
    if not context_chunks:
        return (
            "Пока не нахожу ответа на этот вопрос в загруженных документах. "
            "Уточните вопрос или обратитесь к менеджеру.",
            False,
        )

    context_text = "\n\n---\n\n".join(c["content"] for c in context_chunks)
    user_message = f"Контекст из документов компании:\n{context_text}\n\nВопрос клиента: {question}"

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Без ключа — честно возвращаем заглушку, чтобы демо не падало молча.
        preview = context_chunks[0]["content"][:200]
        return (
            f"[ДЕМО-РЕЖИМ, ANTHROPIC_API_KEY не задан]\n"
            f"Нашёл релевантный фрагмент документа:\n«{preview}...»\n"
            f"С настоящим API-ключом здесь будет связный ответ от Claude.",
            True,
        )

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    answer = "".join(block.text for block in response.content if block.type == "text")
    return answer, True
