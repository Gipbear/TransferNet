"""在训练集同源分组保留折上比较 CWQ 轻量稀疏题型分类器。"""

import argparse
import gc
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

LABELS = ["composition", "conjunction", "comparative", "superlative"]


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def metrics(truth: list[str], predicted: list[str]) -> dict:
    return {"n": len(truth), "correct": int(np.sum(np.asarray(truth) == np.asarray(predicted))),
            "accuracy": accuracy_score(truth, predicted),
            "macro_f1": f1_score(truth, predicted, average="macro"),
            "weighted_f1": f1_score(truth, predicted, average="weighted"),
            "per_class": classification_report(truth, predicted, labels=LABELS, output_dict=True),
            "confusion_order": LABELS, "confusion": confusion_matrix(truth, predicted, labels=LABELS).tolist()}


@dataclass
class FeatureSpec:
    name: str
    char_analyzer: str | None = "char_wb"
    char_max: int = 5
    word_max: int = 0
    word_weight: float = 1.0
    min_df: int = 1

    def build(self) -> FeatureUnion:
        transforms = []
        weights = {}
        if self.char_analyzer:
            transforms.append(("char", TfidfVectorizer(analyzer=self.char_analyzer, ngram_range=(3, self.char_max),
                                                       sublinear_tf=True, min_df=self.min_df, dtype=np.float32)))
            weights["char"] = 1.0
        if self.word_max:
            transforms.append(("word", TfidfVectorizer(ngram_range=(1, self.word_max), sublinear_tf=True,
                                                       min_df=self.min_df, dtype=np.float32)))
            weights["word"] = self.word_weight
        return FeatureUnion(transforms, transformer_weights=weights)


class SparseSearch:
    def __init__(self, source: Path, output: Path) -> None:
        self.source, self.output = source, output
        self.output.mkdir(parents=True, exist_ok=True)
        records = json.loads(source.read_text())
        self.questions = [record["question"] for record in records]
        self.labels = np.asarray([record["compositionality_type"] for record in records])
        self.ids = [record["ID"] for record in records]
        groups = self._groups(records)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=17)
        self.fit_ids, self.val_ids = next(splitter.split(self.questions, self.labels, groups))
        self.train = [self.questions[index] for index in self.fit_ids]
        self.validation = [self.questions[index] for index in self.val_ids]
        assert not {groups[i] for i in self.fit_ids} & {groups[i] for i in self.val_ids}
        assert not {q.strip().casefold() for q in self.train} & {q.strip().casefold() for q in self.validation}
        save_json(output / "split_manifest.json", {
            "source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "seed": 17, "group_field": "webqsp_ID", "n_splits": 5, "selection_fold": 0,
            "train_n": len(self.train), "selection_n": len(self.validation),
            "train_counts": dict(Counter(self.labels[self.fit_ids])),
            "selection_counts": dict(Counter(self.labels[self.val_ids])),
            "train_ids": [self.ids[i] for i in self.fit_ids], "selection_ids": [self.ids[i] for i in self.val_ids],
            "group_overlap_n": 0, "exact_question_overlap_n": 0, "test_used_for_selection": False,
        })
        self.rows = []
        self.best = {}

    @staticmethod
    def _groups(records: list[dict]) -> list[str]:
        parents = {record["webqsp_ID"]: record["webqsp_ID"] for record in records}

        def find(group: str) -> str:
            while parents[group] != group:
                parents[group] = parents[parents[group]]
                group = parents[group]
            return group

        seen = {}
        for record in records:
            question = record["question"].strip().casefold()
            group = record["webqsp_ID"]
            if question in seen:
                parents[find(group)] = find(seen[question])
            seen[question] = group
        return [find(record["webqsp_ID"]) for record in records]

    @staticmethod
    def specifications() -> list[FeatureSpec]:
        return [FeatureSpec("char_wb35"), FeatureSpec("word12", None, word_max=2),
                FeatureSpec("mix12", word_max=2), FeatureSpec("mix12_word_half", word_max=2, word_weight=0.5),
                FeatureSpec("mix12_word_double", word_max=2, word_weight=2.0),
                FeatureSpec("mix13", word_max=3), FeatureSpec("char35", "char"),
                FeatureSpec("mix_char35_word12", "char", word_max=2),
                FeatureSpec("mix12_df2", word_max=2, min_df=2)]

    @staticmethod
    def estimators() -> list[tuple[str, object, dict]]:
        result = []
        for c_value in [0.1, 0.3, 1.0, 3.0, 10.0]:
            result.append(("lr", LogisticRegression(C=c_value, max_iter=1000, random_state=17), {"C": c_value}))
            result.append(("svc", LinearSVC(C=c_value, dual="auto", max_iter=4000, random_state=17), {"C": c_value}))
        result.append(("lr_balanced", LogisticRegression(C=3, class_weight="balanced", max_iter=1000,
                                                        random_state=17), {"C": 3, "class_weight": "balanced"}))
        result.append(("svc_balanced", LinearSVC(C=1, class_weight="balanced", dual="auto", max_iter=4000,
                                               random_state=17), {"C": 1, "class_weight": "balanced"}))
        result.append(("ridge", RidgeClassifier(alpha=1), {"alpha": 1}))
        result.append(("cnb", ComplementNB(alpha=0.1), {"alpha": 0.1}))
        return result

    def run(self) -> None:
        for spec in self.specifications():
            features = spec.build()
            started = perf_counter()
            train_matrix = features.fit_transform(self.train)
            val_matrix = features.transform(self.validation)
            feature_seconds = perf_counter() - started
            for family, estimator, parameters in self.estimators():
                self._candidate(spec, features, train_matrix, val_matrix, family, estimator, parameters,
                                feature_seconds)
            del train_matrix, val_matrix, features
            gc.collect()
        ordered = sorted(self.rows, key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)
        save_json(self.output / "sparse_leaderboard.json", {"candidates": ordered, "winner": ordered[0],
                                                           "selection_scope": "train grouped holdout only"})

    def _candidate(self, spec: FeatureSpec, features: FeatureUnion, train_matrix: sparse.csr_matrix,
                   val_matrix: sparse.csr_matrix, family: str, estimator: object, parameters: dict,
                   feature_seconds: float) -> None:
        started = perf_counter()
        estimator.fit(train_matrix, self.labels[self.fit_ids])
        train_seconds = perf_counter() - started
        started = perf_counter()
        predicted = estimator.predict(val_matrix)
        score = metrics(self.labels[self.val_ids], predicted)
        row = {"feature": spec.__dict__, "family": family, "parameters": parameters, **score,
               "feature_seconds": feature_seconds, "fit_seconds": train_seconds,
               "batch_prediction_seconds": perf_counter() - started, "feature_n": train_matrix.shape[1]}
        self.rows.append(row)
        candidate_id = f"{spec.name}_{family}_{len(self.rows):03d}"
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(val_matrix)
        else:
            probabilities = None
        np.savez_compressed(self.output / f"selection_{candidate_id}.npz", prediction=predicted,
                            decision=estimator.decision_function(val_matrix)
                            if hasattr(estimator, "decision_function") else probabilities,
                            classes=estimator.classes_)
        row["candidate_id"] = candidate_id
        family_key = "svc" if family.startswith("svc") else "lr" if family.startswith("lr") else family
        key = (score["macro_f1"], score["accuracy"])
        if key > self.best.get(family_key, (-1, -1)):
            self.best[family_key] = key
            model_path = self.output / f"best_selection_{family_key}.joblib"
            joblib.dump(Pipeline([("features", features), ("model", estimator)]), model_path, compress=3)
            save_json(self.output / f"best_selection_{family_key}.json", row)
        with (self.output / "sparse_results.jsonl").open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"candidate": candidate_id, "macro_f1": round(score["macro_f1"], 6),
                          "accuracy": round(score["accuracy"], 6), "fit_seconds": round(train_seconds, 2)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/resources/CWQ/origin/ComplexWebQuestions_train.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        SparseSearch(Path(args.train), Path(args.output)).run()


if __name__ == "__main__":
    main()
