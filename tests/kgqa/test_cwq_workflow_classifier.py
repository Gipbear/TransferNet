"""验证离线训练的概率分类器可以按原接口装入冻结模板包。"""

import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

from kgqa.agent.workflow.templates import TemplateLibrary, TextClassifier


class TestFittedTextClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.questions = ["Who is older?", "Which is older?", "Who acts and sings?", "Who writes and sings?"]
        self.labels = ["comparative", "comparative", "conjunction", "conjunction"]
        self.features = FeatureUnion([("words", TfidfVectorizer(ngram_range=(1, 2))),
                                      ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)))])
        self.matrix = self.features.fit_transform(self.questions)
        self.model = LogisticRegression(random_state=17).fit(self.matrix, self.labels)

    def test_fitted_union_preserves_probability_prediction(self) -> None:
        classifier = TextClassifier.from_fitted(self.features, self.model)
        question = "Who acts and sings?"
        probabilities = self.model.predict_proba(self.features.transform([question]))[0]
        index = int(probabilities.argmax())
        self.assertEqual(classifier.predict(question), (str(self.model.classes_[index]), float(probabilities[index])))

    def test_non_probability_classifier_requires_calibration(self) -> None:
        model = LinearSVC().fit(self.matrix, self.labels)
        with self.assertRaisesRegex(TypeError, "predict_proba"):
            TextClassifier.from_fitted(self.features, model)

    def test_unfitted_classifier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TextClassifier.from_fitted(self.features, LogisticRegression())

    def test_calibrated_pipeline_accepts_question_text_directly(self) -> None:
        estimator = Pipeline([("features", self.features), ("model", LinearSVC())])
        model = CalibratedClassifierCV(estimator, cv=2, ensemble=False).fit(self.questions, self.labels)
        classifier = TextClassifier.from_fitted(FunctionTransformer(), model)
        probabilities = model.predict_proba([self.questions[0]])[0]
        index = int(probabilities.argmax())
        self.assertEqual(classifier.predict(self.questions[0]),
                         (str(model.classes_[index]), float(probabilities[index])))

    def test_template_package_roundtrip_preserves_other_resources(self) -> None:
        library = TemplateLibrary()
        library.router = TextClassifier.from_fitted(self.features, self.model)
        library.questions = ["preserved training question"]
        library.patterns = [("preserved relation",)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "templates.joblib"
            joblib.dump(library, path)
            loaded = joblib.load(path)
        self.assertEqual(loaded.questions, library.questions)
        self.assertEqual(loaded.patterns, library.patterns)
        self.assertEqual(loaded.classify(self.questions[0]), library.classify(self.questions[0]))


if __name__ == "__main__":
    unittest.main()
