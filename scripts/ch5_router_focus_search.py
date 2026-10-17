"""围绕领先字符/词特征做小范围扩展，并诊断 MiniLM 混合特征收益。"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from scripts.ch5_router_refine import build_classifier
from scripts.ch5_router_sparse_search import FeatureSpec, SparseSearch, metrics, save_json


def focused_search(root: Path) -> None:
    search = SparseSearch(Path("data/resources/CWQ/origin/ComplexWebQuestions_train.json"), root / "focus")
    specifications = [FeatureSpec("mix_char36_word12", "char", char_max=6, word_max=2),
                      FeatureSpec("mix_char37_word12_df2", "char", char_max=7, word_max=2, min_df=2),
                      FeatureSpec("mix_char36_word12_df2", "char", char_max=6, word_max=2, min_df=2),
                      FeatureSpec("mix_char35_word12_double", "char", word_max=2, word_weight=2),
                      FeatureSpec("mix_char35_word13", "char", word_max=3),
                      FeatureSpec("mix_char35_word12_df2", "char", word_max=2, min_df=2)]
    for spec in specifications:
        features = spec.build()
        started = perf_counter()
        train = features.fit_transform(search.train)
        validation = features.transform(search.validation)
        feature_seconds = perf_counter() - started
        for c_value in [0.3, 1, 3, 10]:
            model = LinearSVC(C=c_value, dual="auto", max_iter=5000, random_state=17)
            search._candidate(spec, features, train, validation, "svc", model, {"C": c_value}, feature_seconds)
        for c_value in [3, 10]:
            model = LogisticRegression(C=c_value, max_iter=1000, random_state=17)
            search._candidate(spec, features, train, validation, "lr", model, {"C": c_value}, feature_seconds)
        for c_value in [1, 3]:
            model = LinearSVC(C=c_value, class_weight="balanced", dual="auto", max_iter=5000, random_state=17)
            search._candidate(spec, features, train, validation, "svc_balanced", model,
                              {"C": c_value, "class_weight": "balanced"}, feature_seconds)
    ordered = sorted(search.rows, key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)
    save_json(search.output / "sparse_leaderboard.json", {"candidates": ordered, "winner": ordered[0]})


def hybrid_search(root: Path) -> None:
    search = SparseSearch(Path("data/resources/CWQ/origin/ComplexWebQuestions_train.json"), root / "hybrid")
    best = json.loads((root / "best_selection_svc.json").read_text())
    features = FeatureSpec(**best["feature"]).build()
    train = features.fit_transform(search.train)
    validation = features.transform(search.validation)
    vectors = np.load(root / "minilm_train_embeddings.npy")
    fit_vectors, val_vectors = vectors[search.fit_ids], vectors[search.val_ids]
    results = []
    best_score = (-1, -1)
    for weight in [0.25, 0.5, 1, 2]:
        fit_matrix = sparse.hstack([train, sparse.csr_matrix(fit_vectors * weight)], format="csr")
        val_matrix = sparse.hstack([validation, sparse.csr_matrix(val_vectors * weight)], format="csr")
        for c_value in [1, 3]:
            estimator = LinearSVC(C=c_value, dual="auto", max_iter=5000, random_state=17)
            started = perf_counter()
            estimator.fit(fit_matrix, search.labels[search.fit_ids])
            predictions = estimator.predict(val_matrix)
            row = {"family": "hybrid_minilm_svc", "feature": best["feature"],
                   "parameters": {"embedding_weight": weight, "C": c_value},
                   **metrics(search.labels[search.val_ids], predictions), "fit_seconds": perf_counter() - started}
            results.append(row)
            if (row["macro_f1"], row["accuracy"]) > best_score:
                best_score = (row["macro_f1"], row["accuracy"])
                joblib.dump({"features": features, "model": estimator}, search.output / "best_selection.joblib")
                save_json(search.output / "best_selection.json", row)
            save_json(search.output / "leaderboard.json", {"candidates": results})
            print(json.dumps({"stage": "hybrid", "parameters": row["parameters"],
                              "macro_f1": row["macro_f1"], "accuracy": row["accuracy"]}), flush=True)


def verify_hybrid(root: Path) -> None:
    search = SparseSearch(Path("data/resources/CWQ/origin/ComplexWebQuestions_train.json"), root / "hybrid")
    records = json.loads(search.source.read_text())
    groups = np.asarray(search._groups(records))
    best = json.loads((search.output / "best_selection.json").read_text())
    vectors = np.load(root / "minilm_train_embeddings.npy")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=17)
    details = []
    for index, (fit_ids, val_ids) in enumerate(splitter.split(search.questions, search.labels, groups)):
        if index > 2:
            break
        features = FeatureSpec(**best["feature"]).build()
        train = features.fit_transform([search.questions[i] for i in fit_ids])
        validation = features.transform([search.questions[i] for i in val_ids])
        weight = best["parameters"]["embedding_weight"]
        train = sparse.hstack([train, sparse.csr_matrix(vectors[fit_ids] * weight)], format="csr")
        validation = sparse.hstack([validation, sparse.csr_matrix(vectors[val_ids] * weight)], format="csr")
        inner = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=29)
                     .split(fit_ids, search.labels[fit_ids], groups[fit_ids]))
        base = LinearSVC(C=best["parameters"]["C"], dual="auto", max_iter=5000, random_state=17)
        model = CalibratedClassifierCV(base, method="sigmoid", cv=inner, ensemble=False).fit(train,
                                                                                          search.labels[fit_ids])
        score = metrics(search.labels[val_ids], model.predict(validation))
        details.append({"fold": index, **score})
        save_json(search.output / "cross_validation.json", {"specification": best, "folds": details,
                  "cv_macro_f1": float(np.mean([row["macro_f1"] for row in details])),
                  "cv_accuracy": float(np.mean([row["accuracy"] for row in details])),
                  "calibration_feature_scope": "outer training fold; all outer validation text excluded"})
        print(json.dumps({"stage": "hybrid_cv", "fold": index, "macro_f1": score["macro_f1"],
                          "accuracy": score["accuracy"]}), flush=True)


def verify_focus(root: Path) -> None:
    search = SparseSearch(Path("data/resources/CWQ/origin/ComplexWebQuestions_train.json"), root / "focus")
    records = json.loads(search.source.read_text())
    groups = np.asarray(search._groups(records))
    best = json.loads((search.output / "best_selection_svc.json").read_text())
    spec = {"family": "cal_svc", "feature": best["feature"], "parameters": best["parameters"]}
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=17)
    details = []
    for index, (fit_ids, val_ids) in enumerate(splitter.split(search.questions, search.labels, groups)):
        if index > 2:
            break
        train = [search.questions[i] for i in fit_ids]
        validation = [search.questions[i] for i in val_ids]
        estimator = build_classifier(spec, list(groups[fit_ids]), search.labels[fit_ids])
        estimator.fit(train, search.labels[fit_ids])
        score = metrics(search.labels[val_ids], estimator.predict(validation))
        details.append({"fold": index, **score})
        save_json(search.output / "cross_validation.json", {"specification": spec, "folds": details,
                  "cv_macro_f1": float(np.mean([row["macro_f1"] for row in details])),
                  "cv_accuracy": float(np.mean([row["accuracy"] for row in details]))})
        print(json.dumps({"stage": "focus_cv", "fold": index, "macro_f1": score["macro_f1"],
                          "accuracy": score["accuracy"]}), flush=True)


def freeze_selection(root: Path) -> None:
    original = json.loads((root / "frozen_selection.json").read_text())
    focus = json.loads((root / "focus/cross_validation.json").read_text())
    hybrid = json.loads((root / "hybrid/cross_validation.json").read_text())
    assert len(focus["folds"]) == len(hybrid["folds"]) == 3
    hybrid_spec = {"family": "hybrid_cal_svc", "feature": hybrid["specification"]["feature"],
                   "parameters": hybrid["specification"]["parameters"]}
    focused_row = {"candidate_id": "focused_cal_svc", "specification": focus["specification"],
                   "cv_macro_f1": focus["cv_macro_f1"], "cv_accuracy": focus["cv_accuracy"]}
    hybrid_row = {"candidate_id": "hybrid_cal_svc", "specification": hybrid_spec,
                  "cv_macro_f1": hybrid["cv_macro_f1"], "cv_accuracy": hybrid["cv_accuracy"]}
    sparse_winner = max([original["winner"], focused_row], key=lambda row: (row["cv_macro_f1"], row["cv_accuracy"]))
    winner = max([sparse_winner, hybrid_row], key=lambda row: (row["cv_macro_f1"], row["cv_accuracy"]))
    frozen = {"winner": winner, "best_sparse": sparse_winner, "baseline": original["baseline"],
              "candidates": [original["winner"], focused_row, hybrid_row], "cv_folds": [0, 1, 2],
              "selection_used_dev": False, "selection_used_test": False,
              "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "selection_rule": "grouped train CV mean Macro-F1 then Accuracy"}
    save_json(root / "final_frozen_selection.json", frozen)
    print(json.dumps({"frozen_winner": winner["candidate_id"], "best_sparse": sparse_winner["candidate_id"],
                      "cv_macro_f1": winner["cv_macro_f1"], "cv_accuracy": winner["cv_accuracy"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["focus", "hybrid", "hybrid_cv", "focus_cv", "freeze"], required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=3):
        if args.stage == "focus":
            focused_search(Path(args.output))
        elif args.stage == "hybrid":
            hybrid_search(Path(args.output))
        elif args.stage == "hybrid_cv":
            verify_hybrid(Path(args.output))
        elif args.stage == "focus_cv":
            verify_focus(Path(args.output))
        else:
            freeze_selection(Path(args.output))


if __name__ == "__main__":
    main()
