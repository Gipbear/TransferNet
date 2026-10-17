"""拟合已冻结的最优路由器，独立评分并导出可接入的模板包。"""

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import fasttext
import joblib
import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from kgqa.agent.workflow.templates import TemplateLibrary, TextClassifier
from kgqa.agent.workflow.text_features import MiniLMTextVectorizer
from scripts.ch5_router_refine import build_classifier
from scripts.ch5_router_sparse_search import FeatureSpec, SparseSearch, metrics, save_json


def predict(classifier: TextClassifier, questions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    probabilities = classifier.model.predict_proba(classifier.vectorizer.transform(questions))
    indices = probabilities.argmax(axis=1)
    return classifier.model.classes_[indices], probabilities.max(axis=1)


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def paired(truth: list[str], baseline: np.ndarray, changed: np.ndarray, groups: list[str]) -> dict:
    before, after = baseline == truth, changed == truth
    corrected = int(np.sum(~before & after))
    harmed = int(np.sum(before & ~after))
    accumulated = defaultdict(lambda: [0, 0])
    for group, delta in zip(groups, after.astype(int) - before.astype(int)):
        accumulated[group][0] += int(delta)
        accumulated[group][1] += 1
    values = np.asarray(list(accumulated.values()))
    draws = np.random.default_rng(17).integers(len(values), size=(3000, len(values)))
    selected = values[draws].sum(axis=1)
    intervals = np.quantile(selected[:, 0] / selected[:, 1], [0.025, 0.975])
    return {"corrected": corrected, "new_errors": harmed, "net": corrected - harmed,
            "accuracy_delta": (corrected - harmed) / len(truth),
            "group_bootstrap_accuracy_delta_ci95": intervals.tolist(), "group_n": len(values)}


class Finalization:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.frozen_path = output / "final_frozen_selection.json"
        self.frozen = json.loads(self.frozen_path.read_text())
        self.train_path = Path("data/resources/CWQ/origin/ComplexWebQuestions_train.json")
        self.training = json.loads(self.train_path.read_text())
        self.questions = [row["question"] for row in self.training]
        self.labels = np.asarray([row["compositionality_type"] for row in self.training])
        self.groups = SparseSearch._groups(self.training)
        self.library_path = Path("data/analysis/20261009_0131__cwq_workflow_implementation/prepared/templates.joblib")
        self.library = TemplateLibrary.load(self.library_path)
        self.models = {"baseline": self.library.router}
        self.report = {"selection": self.frozen, "training_n": len(self.training), "device": "cpu",
                       "frozen_before_dev_test": True, "splits": {}, "resources": {}}

    def run(self) -> None:
        self._fit_frozen()
        self._export_templates()
        self._evaluate_split("dev")
        self._dev_references()
        self._evaluate_split("test")
        self._verify()
        save_json(self.output / "final_report.json", self.report)
        print(json.dumps({"completed": True, "splits": self.report["splits"]}, ensure_ascii=False), flush=True)

    def _fit_frozen(self) -> None:
        for name, spec in [("optimized", self.frozen["winner"]["specification"]),
                           ("sparse", self.frozen["best_sparse"]["specification"])]:
            started = perf_counter()
            if spec["family"] == "hybrid_cal_svc":
                classifier = self._hybrid_classifier(spec)
            else:
                model = build_classifier(spec, self.groups, self.labels).fit(self.questions, self.labels)
                classifier = TextClassifier.from_fitted(FunctionTransformer(), model)
            self.models[name] = classifier
            path = self.output / f"{name}_router.joblib"
            joblib.dump(classifier, path, compress=3)
            self.report["resources"][name] = {**fingerprint(path), "fit_seconds": perf_counter() - started}
            print(json.dumps({"stage": "full_train", "model": name,
                              "fit_seconds": self.report["resources"][name]["fit_seconds"]}), flush=True)

    def _hybrid_classifier(self, spec: dict) -> TextClassifier:
        source = json.loads((self.output / "embedding_model_source.json").read_text())
        sparse_features = FeatureSpec(**spec["feature"]).build().fit(self.questions)
        embedding = MiniLMTextVectorizer(source["snapshot"]).fit(self.questions)
        weight = spec["parameters"]["embedding_weight"]
        features = FeatureUnion([("sparse", sparse_features), ("embedding", embedding)],
                                transformer_weights={"sparse": 1, "embedding": weight})
        vectors = np.load(self.output / "minilm_train_embeddings.npy")
        matrix = sparse.hstack([sparse_features.transform(self.questions),
                               sparse.csr_matrix(vectors * weight)], format="csr")
        folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=29)
                     .split(self.questions, self.labels, self.groups))
        base = LinearSVC(C=spec["parameters"]["C"], dual="auto", max_iter=5000, random_state=17)
        model = CalibratedClassifierCV(base, cv=folds, method="sigmoid", ensemble=False).fit(matrix, self.labels)
        expected = matrix[:10].toarray()
        actual = features.transform(self.questions[:10]).toarray()
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        return TextClassifier.from_fitted(features, model)

    def _export_templates(self) -> None:
        for name in ["optimized", "sparse"]:
            library = copy.copy(self.library)
            library.router = self.models[name]
            path = self.output / f"templates_{name}.joblib"
            library.save(path)
            self.report["resources"][f"templates_{name}"] = fingerprint(path)
        for name in ["optimized", "sparse"]:
            config = json.loads(Path("experiments/configs/ch5/cwq_workflow_v4.json").read_text())
            config["templates"] = str(self.output / f"templates_{name}.joblib")
            config["revision"] = f"router_train_group_cv_{name}"
            config["runs"]["benchmark"][0]["id"] = f"workflow_router_{name}"
            save_json(self.output / f"cwq_workflow_router_{name}.json", config)

    def _evaluate_split(self, split: str) -> None:
        path = Path(f"data/resources/CWQ/origin/ComplexWebQuestions_{split}.json")
        records = json.loads(path.read_text())
        questions = [record["question"] for record in records]
        truth = [record["compositionality_type"] for record in records]
        groups = [record["webqsp_ID"] for record in records]
        evaluations, predictions = {}, {}
        for name, classifier in self.models.items():
            started = perf_counter()
            values, confidence = predict(classifier, questions)
            predictions[name] = values
            evaluations[name] = {**metrics(truth, values), "batch_seconds": perf_counter() - started}
            evaluations[name]["latency"] = self._latency(classifier, questions)
            if name != "baseline":
                evaluations[name]["paired_vs_baseline"] = paired(truth, predictions["baseline"], values, groups)
            target = self.output / f"{split}_{name}_predictions_no_gold.jsonl"
            with target.open("w") as handle:
                for record, kind, score in zip(records, values, confidence):
                    handle.write(json.dumps({"id": record["ID"], "question": record["question"],
                                              "route": {"kind": str(kind), "confidence": float(score)}}) + "\n")
        self.report["splits"][split] = {"source": fingerprint(path), "models": evaluations,
                                       "official_group_overlap_with_train": len(set(groups) & set(self.groups))}
        save_json(self.output / f"{split}_metrics.json", self.report["splits"][split])
        print(json.dumps({"stage": split, "metrics": {name: {"accuracy": row["accuracy"],
                          "macro_f1": row["macro_f1"]} for name, row in evaluations.items()}}), flush=True)

    @staticmethod
    def _latency(classifier: TextClassifier, questions: list[str]) -> dict:
        ids = np.random.default_rng(17).choice(len(questions), size=200, replace=False)
        for index in ids[:5]:
            classifier.predict(questions[index])
        times = []
        for index in ids:
            started = perf_counter()
            classifier.predict(questions[index])
            times.append((perf_counter() - started) * 1000)
        return {"sample_n": len(times), "median_ms": float(np.median(times)),
                "p95_ms": float(np.quantile(times, 0.95)), "cpu_threads": 4, "seed": 17}

    def _dev_references(self) -> None:
        records = json.loads(Path("data/resources/CWQ/origin/ComplexWebQuestions_dev.json").read_text())
        questions = [row["question"] for row in records]
        truth = [row["compositionality_type"] for row in records]
        scores = {}
        for family in ["lr", "svc"]:
            row = json.loads((self.output / f"best_selection_{family}.json").read_text())
            spec = {"family": row["family"], "feature": row["feature"], "parameters": row["parameters"]}
            estimator = build_classifier(spec, self.groups, self.labels).fit(self.questions, self.labels)
            scores[family] = metrics(truth, estimator.predict(questions))
        scores["fasttext"] = self._dev_fasttext(questions, truth)
        scores["minilm"] = self._dev_minilm(questions, truth)
        self.report["dev_references"] = scores
        save_json(self.output / "dev_reference_metrics.json", scores)

    def _dev_fasttext(self, questions: list[str], truth: list[str]) -> dict:
        row = json.loads((self.output / "best_selection_fasttext.json").read_text())
        params = dict(row["parameters"])
        target = self.output / "fasttext_full_train.txt"
        target.write_text("\n".join(f"__label__{kind} {question.lower().replace(chr(10), ' ')}"
                                    for kind, question in zip(self.labels, self.questions)) + "\n")
        model = fasttext.train_supervised(input=str(target), verbose=0, minCount=1, **params)
        labels, _probabilities = model.predict([question.lower().replace("\n", " ") for question in questions])
        model.save_model(str(self.output / "reference_fasttext.bin"))
        return {**metrics(truth, [row[0].removeprefix("__label__") for row in labels]), "parameters": params}

    def _dev_minilm(self, questions: list[str], truth: list[str]) -> dict:
        row = json.loads((self.output / "best_selection_minilm.json").read_text())
        model = SentenceTransformer(row["model_source"]["snapshot"], device="cpu")
        vectors = model.encode(questions, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        train_vectors = np.load(self.output / "minilm_train_embeddings.npy")
        estimator = LogisticRegression(max_iter=1500, random_state=17, **row["parameters"])
        estimator.fit(train_vectors, self.labels)
        joblib.dump(estimator, self.output / "reference_minilm_head.joblib")
        return {**metrics(truth, estimator.predict(vectors)), "parameters": row["parameters"]}

    def _verify(self) -> None:
        base_resources = joblib.hash({key: value for key, value in vars(self.library).items() if key != "router"})
        for name in ["optimized", "sparse"]:
            loaded = TemplateLibrary.load(self.output / f"templates_{name}.joblib")
            others = joblib.hash({key: value for key, value in vars(loaded).items() if key != "router"})
            assert others == base_resources
            for question in self.questions[:20]:
                expected = self.models[name].predict(question)
                actual = loaded.router.predict(question)
                assert actual[0] == expected[0] and abs(actual[1] - expected[1]) < 1e-7
        cached = Path("data/analysis/20261009_0131__cwq_workflow_implementation/dev_routing_predictions_no_gold.jsonl")
        before = {row["id"]: row["route"]["kind"] for row in map(json.loads, cached.read_text().splitlines())}
        after = {row["id"]: row["route"]["kind"] for row in map(json.loads,
                 (self.output / "dev_baseline_predictions_no_gold.jsonl").read_text().splitlines())}
        assert before == after
        self.report["verification"] = {"non_router_resources_unchanged": True, "joblib_roundtrip_n": 20,
                                       "baseline_dev_cached_predictions_identical": True,
                                       "original_templates": fingerprint(self.library_path),
                                       "train_source": fingerprint(self.train_path),
                                       "frozen_selection": fingerprint(self.frozen_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        Finalization(Path(args.output)).run()


if __name__ == "__main__":
    main()
