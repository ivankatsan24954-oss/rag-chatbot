"""
Парсинг загруженных файлов (txt/pdf/docx) и разбивка текста на чанки
с перекрытием — так модель на этапе генерации получает связный контекст
даже если важная мысль оказалась на границе чанка.
"""
import io
from pypdf import PdfReader
from docx import Document as DocxDocument


def parse_file(filename: str, content: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]

    if ext == "txt" or ext == "md":
        return content.decode("utf-8", errors="ignore")

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if ext == "docx":
        doc = DocxDocument(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Неподдерживаемый формат файла: .{ext}")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Разбивка по символам с overlap. Для реального прод-проекта лучше
    резать по предложениям/абзацам (например через nltk), но для MVP
    посимвольная разбивка предсказуема и не требует лишних зависимостей.
    """
    text = " ".join(text.split())  # схлопнуть лишние пробелы/переносы
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # шаг с перекрытием
        if start <= 0:
            break
    return chunks
