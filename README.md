# RAG Chatbot MVP

Простой RAG-чатбот поддержки: загружаете документы компании (txt/pdf/docx),
бот отвечает на вопросы клиентов, опираясь только на их содержимое.

## Структура проекта

```
rag-chatbot/
├── backend/
│   ├── main.py            # FastAPI-приложение, эндпоинты
│   ├── db.py               # SQLite: документы, чанки, лог диалогов
│   ├── ingest.py            # Парсинг файлов + разбивка на чанки
│   ├── retriever.py         # TF-IDF поиск релевантных чанков
│   ├── llm.py                # Обёртка над Anthropic API
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── index.html            # Статический виджет чата (без сборки)
```

## Запуск локально

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# впишите в .env свой ANTHROPIC_API_KEY (можно оставить пустым — тогда бот
# будет работать в демо-режиме, без обращения к Claude)

uvicorn main:app --reload --port 8000
```

Откройте http://localhost:8000 — там будет виджет чата (отдаётся из `../frontend`).

## Переменные окружения

| Переменная | Обязательна | Описание |
|---|---|---|
| `ANTHROPIC_API_KEY` | нет | Без ключа бот вернёт заглушку с найденным фрагментом текста вместо ответа Claude |
