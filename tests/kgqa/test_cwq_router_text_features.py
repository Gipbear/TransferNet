"""验证冻结句向量特征只编码问句，并明确使用 CPU。"""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from kgqa.agent.workflow.text_features import MiniLMTextVectorizer


class TestMiniLMTextVectorizer(unittest.TestCase):
    def test_fit_loads_cpu_encoder_without_supervised_training(self) -> None:
        with patch("kgqa.agent.workflow.text_features.SentenceTransformer") as constructor:
            transformer = MiniLMTextVectorizer("frozen/snapshot").fit(["a"], ["label"])
            constructor.assert_called_once_with("frozen/snapshot", device="cpu")
            constructor.return_value.fit.assert_not_called()
            self.assertIs(transformer.encoder_, constructor.return_value)

    def test_transform_encodes_question_text_as_normalized_sparse_vectors(self) -> None:
        with patch("kgqa.agent.workflow.text_features.SentenceTransformer") as constructor:
            encoder = constructor.return_value
            encoder.encode.return_value = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            transformer = MiniLMTextVectorizer("snapshot").fit(["a", "b"])
            result = transformer.transform(["first question", "second question"])
            encoder.encode.assert_called_once_with(["first question", "second question"], batch_size=64,
                                                   show_progress_bar=False, normalize_embeddings=True)
            np.testing.assert_array_equal(result.toarray(), [[1, 0], [0, 1]])

    def test_clone_does_not_load_an_encoder(self) -> None:
        with patch("kgqa.agent.workflow.text_features.SentenceTransformer") as constructor:
            transformer = clone(MiniLMTextVectorizer("snapshot"))
            self.assertEqual(transformer.model_path, "snapshot")
            constructor.assert_not_called()

    def test_transform_requires_fit(self) -> None:
        with self.assertRaises(NotFittedError):
            MiniLMTextVectorizer("snapshot").transform(["question"])


if __name__ == "__main__":
    unittest.main()
