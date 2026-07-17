"""
Retrieval-слой.

Для MVP используется TF-IDF (scikit-learn) — он работает полностью
локально, без внешних API и без скачивания моделей эмбеддингов,
что удобно для демо и для дешёвого прод-деплоя малому бизнесу.

Для более умного семантического поиска достаточно заменить
`TfidfRetriever` на класс, использующий embeddings API (OpenAI/Anthropic/
Voyage) + векторную БД (pgvector/Qdrant) — интерфейс (`fit`, `search`)
остаётся тем же, остальной код менять не придётся.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self):
        # analyzer='char_wb' + n-граммы 3-5 символов вместо целых слов:
        # для русского языка это критично — TF-IDF по словам не видит
        # связи между "вернуть" и "возврат" из-за словоформ, а по
        # символьным n-граммам общий корень "верну/возвра" ловится.
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_df=0.95,
            min_df=1,
        )
        self.matrix = None
        self.chunks = []  # list[dict]: id, content, document_id, filename

    def fit(self, chunks: list[dict]):
        self.chunks = chunks
        if not chunks:
            self.matrix = None
            return
        texts = [c["content"] for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 4, min_score: float = 0.03):
        if self.matrix is None or not self.chunks:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        results = [
            {**chunk, "score": float(score)}
            for chunk, score in ranked[:top_k]
            if score >= min_score
        ]
        return results
