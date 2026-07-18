"""
Обёртка над Groq API (бесплатный тариф, без привязки карты).
Ключ берётся из переменной окружения GROQ_API_KEY (см. .env.example).
Groq полностью совместим с OpenAI SDK, поэтому используется пакет openai
с указанием base_url на Groq.

Промпт жёстко ограничивает модель контекстом из найденных чанков — если
ответа в документах нет, бот обязан честно сказать, что не знает, а не
выдумывать (без этого RAG-бот быстро теряет доверие клиентов).
"""
import os
from openai import OpenAI

SYSTEM_PROMPT = """Ты — ассистент поддержки компании. Отвечай на вопросы \
клиентов ТОЛЬКО на основе предоставленного контекста из документов компании.

Правила:
1. Если ответа нет в контексте — прямо скажи, что не располагаешь этой \
информацией, и предложи обратиться к менеджеру. Не выдумывай факты.
2. Отвечай кратко и по делу, в дружелюбном тоне.
3. Не упоминай, что ты используешь "контекст" или "документы" — отвечай \
как обычный ассистент компании.
"""

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        _client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    return _client


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

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        # Без ключа — честно возвращаем заглушку, чтобы демо не падало молча.
        preview = context_chunks[0]["content"][:200]
        return (
            f"[ДЕМО-РЕЖИМ, GROQ_API_KEY не задан]\n"
            f"Нашёл релевантный фрагмент документа:\n«{preview}...»\n"
            f"С настоящим API-ключом здесь будет связный ответ от модели.",
            True,
        )

    client = _get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    answer = response.choices[0].message.content
    return answer, True
