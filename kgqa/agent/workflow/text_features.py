"""提供 CPU 上冻结的 MiniLM 问句向量，供离线分类器与运行时复用。"""

from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MiniLMTextVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def fit(self, questions: list[str], labels: list[str] | None = None) -> "MiniLMTextVectorizer":
        self.encoder_ = SentenceTransformer(self.model_path, device="cpu")
        return self

    def transform(self, questions: list[str]) -> sparse.csr_matrix:
        check_is_fitted(self, ["encoder_"])
        vectors = self.encoder_.encode(list(questions), batch_size=64, show_progress_bar=False,
                                       normalize_embeddings=True)
        return sparse.csr_matrix(vectors)
