"""在同一训练保留折上比较 fastText 与冻结 MiniLM 的 CPU 分类效果。"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import fasttext
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from scripts.ch5_router_sparse_search import LABELS, metrics, save_json


class OptionalSearch:
    def __init__(self, output: Path) -> None:
        self.output = output
        records = json.loads(Path("data/resources/CWQ/origin/ComplexWebQuestions_train.json").read_text())
        split = json.loads((output / "split_manifest.json").read_text())
        by_id = {record["ID"]: record for record in records}
        self.training = [by_id[key] for key in split["train_ids"]]
        self.validation = [by_id[key] for key in split["selection_ids"]]
        self.all_records = records
        self.labels = [record["compositionality_type"] for record in self.training]
        self.truth = [record["compositionality_type"] for record in self.validation]

    def fasttext(self) -> None:
        train_file = self.output / "fasttext_selection_train.txt"
        train_file.write_text("\n".join(self._line(record) for record in self.training) + "\n")
        questions = [record["question"].lower().replace("\n", " ") for record in self.validation]
        results = []
        best = (-1, -1)
        configurations = []
        for dimension in [50, 100]:
            for grams in [1, 2, 3]:
                configurations.extend([(dimension, grams, 0.3, 25), (dimension, grams, 0.7, 50)])
        for index, (dimension, grams, rate, epoch) in enumerate(configurations):
            started = perf_counter()
            model = fasttext.train_supervised(input=str(train_file), dim=dimension, wordNgrams=grams,
                                              lr=rate, epoch=epoch, thread=4, bucket=100000, minCount=1,
                                              loss="softmax", seed=17, verbose=0)
            train_seconds = perf_counter() - started
            started = perf_counter()
            labels, probabilities = model.predict(questions, k=4)
            predictions = [row[0].removeprefix("__label__") for row in labels]
            row = {"family": "fasttext", "candidate_id": f"fasttext_{index:03d}",
                   "parameters": {"dim": dimension, "wordNgrams": grams, "lr": rate, "epoch": epoch,
                                  "bucket": 100000, "thread": 4, "seed": 17, "loss": "softmax"},
                   **metrics(self.truth, predictions), "fit_seconds": train_seconds,
                   "batch_prediction_seconds": perf_counter() - started}
            aligned = np.zeros((len(questions), len(LABELS)))
            for row_index, (names, values) in enumerate(zip(labels, probabilities)):
                for name, value in zip(names, values):
                    aligned[row_index, LABELS.index(name.removeprefix("__label__"))] = value
            np.savez_compressed(self.output / f"selection_{row['candidate_id']}.npz", prediction=predictions,
                                decision=aligned, classes=LABELS)
            key = (row["macro_f1"], row["accuracy"])
            if key > best:
                best = key
                model.save_model(str(self.output / "best_selection_fasttext.bin"))
                save_json(self.output / "best_selection_fasttext.json", row)
            results.append(row)
            save_json(self.output / "fasttext_leaderboard.json", {"candidates": results})
            print(json.dumps({"candidate": row["candidate_id"], "macro_f1": row["macro_f1"],
                              "accuracy": row["accuracy"], "fit_seconds": train_seconds}), flush=True)

    @staticmethod
    def _line(record: dict) -> str:
        question = record["question"].lower().replace("\n", " ")
        return f"__label__{record['compositionality_type']} {question}"

    def embedding(self) -> None:
        source = json.loads((self.output / "embedding_model_source.json").read_text())
        model = SentenceTransformer(source["snapshot"], device="cpu")
        target = self.output / "minilm_train_embeddings.npy"
        started = perf_counter()
        if target.exists():
            vectors = np.load(target)
        else:
            vectors = model.encode([record["question"] for record in self.all_records], batch_size=64,
                                   show_progress_bar=False, normalize_embeddings=True)
            np.save(target, vectors)
        print(json.dumps({"embedding_seconds": perf_counter() - started, "shape": list(vectors.shape)}), flush=True)
        id_to_index = {record["ID"]: index for index, record in enumerate(self.all_records)}
        train = vectors[[id_to_index[record["ID"]] for record in self.training]]
        validation = vectors[[id_to_index[record["ID"]] for record in self.validation]]
        results = []
        best = (-1, -1)
        for c_value in [0.1, 1, 10, 100]:
            for family in ["lr", "svc"]:
                if family == "lr":
                    estimator = LogisticRegression(C=c_value, max_iter=1500, random_state=17)
                else:
                    estimator = LinearSVC(C=c_value, dual="auto", max_iter=5000, random_state=17)
                started = perf_counter()
                estimator.fit(train, self.labels)
                train_seconds = perf_counter() - started
                predictions = estimator.predict(validation)
                row = {"family": f"minilm_{family}", "candidate_id": f"minilm_{family}_C{c_value}",
                       "parameters": {"C": c_value}, **metrics(self.truth, predictions),
                       "fit_seconds": train_seconds, "model_source": source}
                np.savez_compressed(self.output / f"selection_{row['candidate_id']}.npz", prediction=predictions,
                                    decision=estimator.decision_function(validation), classes=estimator.classes_)
                results.append(row)
                if (row["macro_f1"], row["accuracy"]) > best:
                    best = (row["macro_f1"], row["accuracy"])
                    joblib.dump(estimator, self.output / "best_selection_minilm.joblib")
                    save_json(self.output / "best_selection_minilm.json", row)
                save_json(self.output / "embedding_leaderboard.json", {"candidates": results})
                print(json.dumps({"candidate": row["candidate_id"], "macro_f1": row["macro_f1"],
                                  "accuracy": row["accuracy"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", choices=["fasttext", "embedding"], required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        search = OptionalSearch(Path(args.output))
        getattr(search, args.family)()


if __name__ == "__main__":
    main()
