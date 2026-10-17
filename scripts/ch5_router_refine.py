"""复核训练保留折领先模型、概率校准及轻量集成，冻结最终分类方案。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from scripts.ch5_router_sparse_search import FeatureSpec, SparseSearch, metrics, save_json


def build_classifier(spec: dict, groups: list[str], labels: np.ndarray) -> object:
    family = spec["family"]
    params = spec["parameters"]
    if family == "ensemble":
        estimators = []
        for index, member in enumerate(spec["members"]):
            estimators.append((f"member{index}", build_classifier(member, groups, labels)))
        return VotingClassifier(estimators, voting="soft", weights=spec["weights"])
    features = FeatureSpec(**spec["feature"]).build()
    if family.startswith("lr"):
        estimator = LogisticRegression(max_iter=1000, random_state=17, **params)
    elif family.startswith("svc") or family == "cal_svc":
        estimator = LinearSVC(dual="auto", max_iter=4000, random_state=17, **params)
    elif family == "cal_ridge":
        estimator = RidgeClassifier(**params)
    else:
        raise ValueError(f"未支持的冻结模型: {family}")
    pipeline = Pipeline([("features", features), ("model", estimator)])
    if family.startswith("cal_"):
        folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=29).split(labels, labels, groups))
        return CalibratedClassifierCV(pipeline, method="sigmoid", cv=folds, ensemble=False, n_jobs=1)
    return pipeline


class Refinement:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.search = SparseSearch(Path("data/resources/CWQ/origin/ComplexWebQuestions_train.json"), output)
        records = json.loads(self.search.source.read_text())
        self.groups = np.asarray(self.search._groups(records))
        self.rows = []

    @staticmethod
    def specification(row: dict, family: str | None = None) -> dict:
        return {"feature": row["feature"], "family": family or row["family"], "parameters": row["parameters"]}

    def run(self) -> None:
        sparse_rows = json.loads((self.output / "sparse_leaderboard.json").read_text())["candidates"]
        baseline = next(row for row in sparse_rows if row["candidate_id"] == "char_wb35_lr_005")
        best_lr = max((row for row in sparse_rows if row["family"].startswith("lr")), key=lambda r: r["macro_f1"])
        svm_rows = sorted((row for row in sparse_rows if row["family"].startswith("svc")),
                          key=lambda r: r["macro_f1"], reverse=True)
        chosen_svm = self._different_features(svm_rows, 2)
        best_ridge = max((row for row in sparse_rows if row["family"] == "ridge"), key=lambda r: r["macro_f1"])
        starters = [("baseline", self.specification(baseline)), ("best_lr", self.specification(best_lr)),
                    ("cal_svc1", self.specification(chosen_svm[0], "cal_svc")),
                    ("cal_svc2", self.specification(chosen_svm[1], "cal_svc")),
                    ("cal_ridge", self.specification(best_ridge, "cal_ridge"))]
        for name, spec in starters:
            self._selection_candidate(name, spec)
        for weight in [[1, 1], [1, 2], [2, 1]]:
            spec = {"family": "ensemble", "parameters": {}, "weights": weight,
                    "members": [starters[1][1], starters[2][1]]}
            self._selection_candidate(f"ensemble_lr_svc_{weight[0]}_{weight[1]}", spec)
        spec = {"family": "ensemble", "parameters": {}, "weights": [1, 1],
                "members": [starters[2][1], starters[3][1]]}
        self._selection_candidate("ensemble_svc_svc", spec)
        shortlist = sorted(self.rows[1:], key=lambda r: (r["macro_f1"], r["accuracy"]), reverse=True)[:3]
        self._cross_validation([self.rows[0], *shortlist])

    @staticmethod
    def _different_features(rows: list[dict], limit: int) -> list[dict]:
        seen, selected = set(), []
        for row in rows:
            name = row["feature"]["name"]
            if name not in seen:
                selected.append(row)
                seen.add(name)
            if len(selected) == limit:
                break
        return selected

    def _selection_candidate(self, name: str, spec: dict) -> None:
        fit_ids, val_ids = self.search.fit_ids, self.search.val_ids
        estimator = build_classifier(spec, list(self.groups[fit_ids]), self.search.labels[fit_ids])
        started = perf_counter()
        estimator.fit(self.search.train, self.search.labels[fit_ids])
        predictions = estimator.predict(self.search.validation)
        row = {"candidate_id": name, "specification": spec,
               **metrics(self.search.labels[val_ids], predictions), "fit_seconds": perf_counter() - started}
        probabilities = estimator.predict_proba(self.search.validation)
        np.savez_compressed(self.output / f"selection_refine_{name}.npz", prediction=predictions,
                            probabilities=probabilities, classes=estimator.classes_)
        joblib.dump(estimator, self.output / f"selection_refine_{name}.joblib", compress=3)
        self.rows.append(row)
        save_json(self.output / "refinement_leaderboard.json", {"candidates": self.rows})
        print(json.dumps({"stage": "selection", "candidate": name, "macro_f1": row["macro_f1"],
                          "accuracy": row["accuracy"], "fit_seconds": row["fit_seconds"]}), flush=True)

    def _cross_validation(self, candidates: list[dict]) -> None:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=17)
        folds = list(splitter.split(self.search.questions, self.search.labels, self.groups))
        for row in candidates:
            scores = [row["macro_f1"]]
            accuracies = [row["accuracy"]]
            details = []
            for index in [1, 2]:
                fit_ids, val_ids = folds[index]
                estimator = build_classifier(row["specification"], list(self.groups[fit_ids]),
                                              self.search.labels[fit_ids])
                train = [self.search.questions[i] for i in fit_ids]
                validation = [self.search.questions[i] for i in val_ids]
                estimator.fit(train, self.search.labels[fit_ids])
                result = metrics(self.search.labels[val_ids], estimator.predict(validation))
                details.append({"fold": index, **result})
                scores.append(result["macro_f1"])
                accuracies.append(result["accuracy"])
                print(json.dumps({"stage": "cross_validation", "candidate": row["candidate_id"],
                                  "fold": index, "macro_f1": result["macro_f1"],
                                  "accuracy": result["accuracy"]}), flush=True)
            row.update({"cv_macro_f1": float(np.mean(scores)), "cv_accuracy": float(np.mean(accuracies)),
                        "cv_macro_f1_std": float(np.std(scores)), "cv_folds": [0, 1, 2], "cv_details": details})
            save_json(self.output / "cross_validation.json", {"candidates": candidates})
        winner = max(candidates[1:], key=lambda row: (row["cv_macro_f1"], row["cv_accuracy"]))
        save_json(self.output / "frozen_selection.json", {"winner": winner, "baseline": candidates[0],
                                                         "selection_used_dev": False, "selection_used_test": False,
                                                         "ranking": "grouped train CV mean Macro-F1 then Accuracy"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        Refinement(Path(args.output)).run()


if __name__ == "__main__":
    main()
