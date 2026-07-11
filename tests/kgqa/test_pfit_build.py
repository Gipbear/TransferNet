"""pfit.build:输入契约、与 llm_infer build 的逐条 parity、分层采样、manifest。"""
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
# legacy 脚本用同目录裸导入(from kg_format import ...),需把 llm_infer 加入 sys.path
sys.path.insert(0, str(_PROJECT / "llm_infer"))
import build_kgcot_dataset as legacy_build  # noqa: E402


def _fake_records(n=30, seed=7, with_golden=True, hops=(1, 2, 3)):
    """构造 retrieve 输出形态的记录:mmr_reason_paths + golden + topics + hop。"""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        gold = f"m.gold{i}"
        paths = [{"path": [[f"m.topic{i}", f"rel.a.b{j}", gold if j == 0 else f"m.tail{i}_{j}"]],
                  "log_score": -0.1 * (j + 1)}
                 for j in range(rng.randint(2, 5))]
        rec = {
            "sample_index": i,
            "question": f"who is entity number {i}",
            "topics": [f"m.topic{i}"],
            "hop": hops[i % len(hops)],
            "mmr_reason_paths": paths,
            "prediction": {gold: 0.9},
        }
        if with_golden:
            rec["golden"] = [gold]
        records.append(rec)
    return records


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _emap_file(dirpath, records):
    """为记录中所有实体生成 MID→Name 映射文件。"""
    path = os.path.join(dirpath, "emap.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            for p in r["mmr_reason_paths"]:
                for h, _, t in p["path"]:
                    for mid in (h, t):
                        f.write(f"{mid}\tName[{mid}]\n")
    return path


class TestInputContract(unittest.TestCase):
    def test_missing_golden_raises_with_hint(self):
        from kgqa.pfit import build as pfit_build
        with tempfile.TemporaryDirectory() as d:
            inp = os.path.join(d, "in.jsonl")
            _write_jsonl(inp, _fake_records(3, with_golden=False))
            with self.assertRaises(ValueError) as ctx:
                pfit_build.run_build(dataset="webqsp", input_path=inp,
                                     exp_dir=os.path.join(d, "exp"), fmt="v2")
            self.assertIn("golden", str(ctx.exception))
            self.assertIn("retrieve", str(ctx.exception))


class TestLegacyParity(unittest.TestCase):
    """同输入 + 同配置 + 同 seed:pfit 与 legacy 产物 messages 逐条一致。"""

    def _run_pair(self, records, *, fmt, path_format, entity_map_path=None,
                  shuffle=True, show_score=False, distractor_ratio=None,
                  sample_n=0, seed=42):
        from kgqa.pfit import build as pfit_build
        with tempfile.TemporaryDirectory() as d:
            inp = os.path.join(d, "in.jsonl")
            _write_jsonl(inp, records)

            legacy_out = os.path.join(d, "legacy.jsonl")
            emap = legacy_build.load_entity_map(entity_map_path) if entity_map_path else None
            import logging
            legacy_build.build(
                inp, legacy_out, fmt, shuffle, distractor_ratio, sample_n,
                show_score, random.Random(seed), logging.getLogger("t"),
                path_format=path_format, entity_map=emap,
            )

            exp_dir = os.path.join(d, "exp")
            out_path = pfit_build.run_build(
                dataset="webqsp", input_path=inp, exp_dir=exp_dir, fmt=fmt,
                path_format=path_format,
                entity_repr="name" if entity_map_path else "mid",
                entity_map_path=entity_map_path,
                shuffle=shuffle, show_score=show_score,
                distractor_ratio=distractor_ratio, sample_n=sample_n, seed=seed,
            )

            with open(legacy_out, encoding="utf-8") as f:
                legacy_samples = [json.loads(l) for l in f]
            with open(out_path, encoding="utf-8") as f:
                pfit_samples = [json.loads(l) for l in f]
        return legacy_samples, pfit_samples

    def _assert_messages_equal(self, legacy_samples, pfit_samples):
        self.assertEqual(len(legacy_samples), len(pfit_samples))
        self.assertGreater(len(legacy_samples), 0)
        for i, (a, b) in enumerate(zip(legacy_samples, pfit_samples)):
            self.assertEqual(a["messages"], b["messages"], f"messages 不一致 @ {i}")

    def test_chain_name_v2_main_config(self):
        records = _fake_records(20)
        with tempfile.TemporaryDirectory() as d:
            emap_path = _emap_file(d, records)
            legacy_samples, pfit_samples = self._run_pair(
                records, fmt="v2", path_format="chain", entity_map_path=emap_path)
            self._assert_messages_equal(legacy_samples, pfit_samples)

    def test_other_formats_mid(self):
        records = _fake_records(8)
        for fmt, path_format in [("v1", "arrow"), ("v3", "tuple"), ("v4", "nl")]:
            with self.subTest(fmt=fmt, path_format=path_format):
                legacy_samples, pfit_samples = self._run_pair(
                    records, fmt=fmt, path_format=path_format)
                self._assert_messages_equal(legacy_samples, pfit_samples)

    def test_plain_sampling_parity(self):
        records = _fake_records(20)
        legacy_samples, pfit_samples = self._run_pair(
            records, fmt="v2", path_format="chain", sample_n=10)
        self._assert_messages_equal(legacy_samples, pfit_samples)


class TestStratifiedSampling(unittest.TestCase):
    def test_hop_proportions_follow_source(self):
        from kgqa.pfit import build as pfit_build
        # 60 条:hop1/2/3 各 30/20/10
        records = ([r for r in _fake_records(30, hops=(1,))]
                   + [r for r in _fake_records(20, seed=8, hops=(2,))]
                   + [r for r in _fake_records(10, seed=9, hops=(3,))])
        with tempfile.TemporaryDirectory() as d:
            inp = os.path.join(d, "in.jsonl")
            _write_jsonl(inp, records)
            out_path = pfit_build.run_build(
                dataset="metaqa", input_path=inp,
                exp_dir=os.path.join(d, "exp"), fmt="v2", path_format="chain",
                sample_n=30, stratify_by_hop=True, seed=42)
            with open(out_path, encoding="utf-8") as f:
                samples = [json.loads(l) for l in f]
        hops = [s["_meta"]["hop"] for s in samples]
        self.assertEqual(len(samples), 30)
        self.assertEqual(hops.count(1), 15)
        self.assertEqual(hops.count(2), 10)
        self.assertEqual(hops.count(3), 5)

    def test_metaqa_uses_name_prompt_and_fills_topic(self):
        from kgqa.pfit import build as pfit_build
        records = [{
            "sample_index": 0, "question": "what does E_S appear in",
            "topics": ["Grégoire Colin"], "hop": 1,
            "golden": ["Before the Rain"],
            "mmr_reason_paths": [{"path": [["Grégoire Colin", "starred_actors_reverse",
                                            "Before the Rain"]], "log_score": -0.1}],
            "prediction": {},
        }]
        with tempfile.TemporaryDirectory() as d:
            inp = os.path.join(d, "in.jsonl")
            _write_jsonl(inp, records)
            out_path = pfit_build.run_build(
                dataset="metaqa", input_path=inp,
                exp_dir=os.path.join(d, "exp"), fmt="v2", path_format="chain")
            with open(out_path, encoding="utf-8") as f:
                sample = json.loads(f.readline())
        # MetaQA 天然 name:即便无映射文件,system prompt 也必须是 name 措辞
        self.assertIn("entity names", sample["messages"][0]["content"])
        self.assertIn("Grégoire Colin appear in", sample["messages"][1]["content"])
        self.assertNotIn("E_S", sample["messages"][1]["content"])

    def test_metaqa_rejection_flag_rejected(self):
        from kgqa.pfit import build as pfit_build
        with tempfile.TemporaryDirectory() as d:
            inp = os.path.join(d, "in.jsonl")
            _write_jsonl(inp, _fake_records(3))
            with self.assertRaises(ValueError):
                pfit_build.run_build(dataset="metaqa", input_path=inp,
                                     exp_dir=os.path.join(d, "exp"), fmt="v2",
                                     include_rejection=True)


class TestManifest(unittest.TestCase):
    def test_skip_on_same_config_error_on_change(self):
        from kgqa.pfit import build as pfit_build
        records = _fake_records(5)
        with tempfile.TemporaryDirectory() as d:
            inp = os.path.join(d, "in.jsonl")
            _write_jsonl(inp, records)
            exp_dir = os.path.join(d, "exp")
            out1 = pfit_build.run_build(dataset="webqsp", input_path=inp,
                                        exp_dir=exp_dir, fmt="v2", path_format="chain")
            mtime = os.path.getmtime(out1)
            # 同配置重跑:跳过(输出不重写)
            out2 = pfit_build.run_build(dataset="webqsp", input_path=inp,
                                        exp_dir=exp_dir, fmt="v2", path_format="chain")
            self.assertEqual(os.path.getmtime(out2), mtime)
            # 改配置重跑:报不一致
            with self.assertRaises(RuntimeError):
                pfit_build.run_build(dataset="webqsp", input_path=inp,
                                     exp_dir=exp_dir, fmt="v3", path_format="chain")
            # manifest 落盘且含 build 节
            with open(os.path.join(exp_dir, "manifest.json"), encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertIn("build", manifest)
            self.assertIn("config", manifest["build"])
            self.assertIn("inputs", manifest["build"])


if __name__ == "__main__":
    unittest.main()
