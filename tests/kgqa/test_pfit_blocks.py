"""分块构样测试（P2 beam40 分段拒答）。

同一题的候选池按排名切块，每块单独成样本：块内有 gold 尾匹配给「引用+答案」，
整块无匹配给「Supporting Paths: None / Answer: None」。
三组对照（A 自然拒答 / B 排名分块 / C 合成拒答）必须样本量与回答-拒答占比可控，
且同题各块共享 group 键、训练时不得跨 train/val。
"""
import json
import unittest

from kgqa.pfit.blocks import (
    BlockBuilder,
    BlockSpec,
    block_stats,
    build_group_samples,
    grouped_split_indices,
)
from kgqa.pfit.build import output_v2_reject

REJECTION = output_v2_reject()
# clean_question 直接用原问题，测试不关心数据集特有的清洗规则
def IDENTITY_CLEAN(question, topics):
    return question


def make_path(tail: str, rel: str = "r") -> dict:
    return {"path": [["m.head", rel, tail]], "log_score": -1.0}


def make_record(index: int, n_paths: int, gold_positions: list[int],
                extra_tails: bool = True) -> dict:
    """构造一条检索记录：gold_positions 是（1-based）尾实体命中 golden 的路径序号。"""
    paths = []
    gold = []
    for i in range(1, n_paths + 1):
        tail = f"m.gold{i}" if i in gold_positions else f"m.dist{i}"
        paths.append(make_path(tail))
        if i in gold_positions:
            gold.append(tail)
    record = {"sample_index": index, "question": f"question {index}",
              "topics": ["m.head"], "hop": 2, "golden": gold,
              "mmr_reason_paths": paths}
    if not extra_tails:
        record.pop("golden")
    return record


def make_builder(**kwargs) -> BlockBuilder:
    spec = kwargs.pop("spec", BlockSpec(block_size=20, max_blocks=2))
    defaults = {"fmt": "v2", "shuffle": True, "distractor_ratio": None, "show_score": False,
                "path_format": "chain", "use_entity_names": True, "entity_map": None,
                "clean_question": IDENTITY_CLEAN, "seed": 17}
    defaults.update(kwargs)
    return BlockBuilder(spec, **defaults)


def cited_indices(sample: dict) -> list[int]:
    line = sample["messages"][2]["content"].splitlines()[0]
    payload = line.replace("Supporting Paths:", "").strip()
    return [] if payload == "None" else [int(x) for x in payload.split(",")]


def user_path_indices(sample: dict) -> list[int]:
    """user 消息里出现的路径编号（按 'N: ' 行首取出）。"""
    found = []
    for line in sample["messages"][1]["content"].splitlines():
        head = line.split(":", 1)[0].strip()
        if head.isdigit():
            found.append(int(head))
    return found


class TestBlockSplit(unittest.TestCase):
    def test_full_two_blocks(self):
        builder = make_builder()
        blocks = builder.split(make_record(0, 40, [3]))
        self.assertEqual([b[0] for b in blocks], [1, 2])
        self.assertEqual([len(b[1]) for b in blocks], [20, 20])

    def test_short_second_block(self):
        blocks = make_builder().split(make_record(0, 25, [3]))
        self.assertEqual([len(b[1]) for b in blocks], [20, 5])

    def test_exactly_one_block(self):
        blocks = make_builder().split(make_record(0, 20, [3]))
        self.assertEqual([len(b[1]) for b in blocks], [20])

    def test_single_short_block(self):
        blocks = make_builder().split(make_record(0, 4, [3]))
        self.assertEqual([len(b[1]) for b in blocks], [4])

    def test_empty_pool(self):
        self.assertEqual(make_builder().split(make_record(0, 0, [])), [])

    def test_block_size_is_configurable(self):
        builder = make_builder(spec=BlockSpec(block_size=5, max_blocks=3))
        blocks = builder.split(make_record(0, 12, [3]))
        self.assertEqual([len(b[1]) for b in blocks], [5, 5, 2])


class TestBlockSamples(unittest.TestCase):
    def test_answer_block_cites_indices_within_block(self):
        record = make_record(7, 40, [25])
        block2 = make_builder().split(record)[1]
        sample = make_builder().build_block(record, block2[0], block2[1])
        cited = cited_indices(sample)
        self.assertEqual(len(cited), 1)
        self.assertTrue(all(1 <= i <= 20 for i in cited))
        self.assertEqual(user_path_indices(sample), list(range(1, 21)))
        self.assertFalse(sample["_meta"]["is_rejection"])

    def test_cited_index_points_at_golden_path(self):
        record = make_record(7, 40, [25])
        builder = make_builder(shuffle=False)
        block2 = builder.split(record)[1]
        sample = builder.build_block(record, block2[0], block2[1])
        idx = cited_indices(sample)[0]
        self.assertIn("m.gold25", sample["messages"][1]["content"].split("\n")[idx + 2])

    def test_answer_is_path_tail(self):
        record = make_record(7, 40, [25])
        block2 = make_builder().split(record)[1]
        sample = make_builder().build_block(record, block2[0], block2[1])
        self.assertEqual(sample["messages"][2]["content"].splitlines()[1], "Answer: m.gold25")

    def test_block_without_golden_is_rejection(self):
        record = make_record(7, 40, [3])
        block2 = make_builder().split(record)[1]
        sample = make_builder().build_block(record, block2[0], block2[1])
        self.assertEqual(sample["messages"][2]["content"], REJECTION)
        self.assertTrue(sample["_meta"]["is_rejection"])
        self.assertEqual(cited_indices(sample), [])

    def test_shuffle_keeps_path_set_and_renumbers(self):
        record = make_record(7, 40, [25])
        builder = make_builder(shuffle=True)
        block2 = builder.split(record)[1]
        sample = builder.build_block(record, block2[0], block2[1])
        self.assertEqual(user_path_indices(sample), list(range(1, 21)))
        for tail in ("m.gold25", "m.dist21", "m.dist40"):
            self.assertIn(tail, sample["messages"][1]["content"])

    def test_meta_records_group_and_block(self):
        record = make_record(11, 40, [25])
        builder = make_builder()
        blocks = builder.split(record)
        first = builder.build_block(record, blocks[0][0], blocks[0][1])
        second = builder.build_block(record, blocks[1][0], blocks[1][1])
        self.assertEqual(first["_meta"]["group"], 11)
        self.assertEqual(second["_meta"]["group"], 11)
        self.assertEqual((first["_meta"]["block"], second["_meta"]["block"]), (1, 2))

    def test_block_sample_never_exceeds_block_size_paths(self):
        record = make_record(7, 40, [25])
        builder = make_builder()
        for block_index, paths in builder.split(record):
            sample = builder.build_block(record, block_index, paths)
            self.assertLessEqual(len(user_path_indices(sample)), 20)

    def test_empty_golden_record_skipped(self):
        record = make_record(7, 40, [])
        self.assertIsNone(make_builder().build_block(record, 1, record["mmr_reason_paths"][:20]))


class TestNaturalGroup(unittest.TestCase):
    """A 组：只用第一块的自然拒答基线。"""

    def test_only_first_block(self):
        records = [make_record(0, 40, [3]), make_record(1, 40, [25])]
        samples = make_builder().build_natural(records)
        self.assertEqual(len(samples), 2)
        self.assertTrue(all(s["_meta"]["block"] == 1 for s in samples))
        self.assertEqual(sum(s["_meta"]["is_rejection"] for s in samples), 1)

    def test_rejection_when_no_golden_tail(self):
        """golden 非空但没有任何路径尾命中它——这才是「检索没抓到」。"""
        record = make_record(0, 5, [])
        record["golden"] = ["m.elsewhere"]
        samples = make_builder().build_natural([record])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["messages"][2]["content"], REJECTION)
        self.assertTrue(samples[0]["_meta"]["is_rejection"])

    def test_record_without_golden_field_skipped(self):
        record = make_record(0, 5, [])
        self.assertEqual(make_builder().build_natural([record]), [])


class TestBlocksGroup(unittest.TestCase):
    """B 组：A + 第二块样本。"""

    def setUp(self):
        self.records = [make_record(0, 40, [3]), make_record(1, 40, [25]),
                        make_record(2, 25, [22]), make_record(3, 20, [5])]

    def test_extra_samples_are_second_blocks_only(self):
        builder = make_builder()
        natural = builder.build_natural(self.records)
        blocks = builder.build_blocks(self.records)
        self.assertEqual(len(natural), 4)
        # 40/40/25 有第二块，20 条的没有
        self.assertEqual(len(blocks), 4 + 3)
        extras = [s for s in blocks if s["_meta"]["block"] == 2]
        self.assertEqual(len(extras), 3)
        self.assertEqual([len(user_path_indices(s)) for s in extras], [20, 20, 5])

    def test_extra_rejection_share(self):
        """r0 的 gold 在第 3 条（第一块）⇒ 第二块整块无命中，是天然的块级拒答监督。"""
        blocks = make_builder().build_blocks(self.records)
        extras = [s for s in blocks if s["_meta"]["block"] == 2]
        self.assertEqual(sum(s["_meta"]["is_rejection"] for s in extras), 1)

    def test_base_samples_identical_to_natural(self):
        builder = make_builder()
        natural = builder.build_natural(self.records)
        blocks = builder.build_blocks(self.records)
        first = [s for s in blocks if s["_meta"]["block"] == 1]
        self.assertEqual([s["messages"] for s in first], [s["messages"] for s in natural])


class TestSyntheticGroup(unittest.TestCase):
    """C 组：额外样本量与回答-拒答占比与 B 组逐项对齐。"""

    def setUp(self):
        # 一半题的正例在第一块、一半在第二块 ⇒ B 组额外样本里回答/拒答各占一半
        pattern = [[3], [25], [], [25]]
        self.records = [make_record(i, 40, pattern[i % 4]) for i in range(12)]
        self.builder = make_builder()
        self.blocks = self.builder.build_blocks(self.records)
        self.extras_b = [s for s in self.blocks if s["_meta"]["block"] == 2]

    def test_extra_count_matches(self):
        extras_c = self.builder.build_synthetic_extras(self.records, self.extras_b)
        self.assertEqual(len(extras_c), len(self.extras_b))

    def test_extra_rejection_count_matches(self):
        extras_c = self.builder.build_synthetic_extras(self.records, self.extras_b)
        want = sum(s["_meta"]["is_rejection"] for s in self.extras_b)
        self.assertEqual(sum(s["_meta"]["is_rejection"] for s in extras_c), want)

    def test_synthetic_rejections_keep_no_golden_path(self):
        extras_c = self.builder.build_synthetic_extras(self.records, self.extras_b)
        for sample in extras_c:
            if not sample["_meta"]["is_rejection"]:
                continue
            self.assertEqual(sample["messages"][2]["content"], REJECTION)
            user = sample["messages"][1]["content"]
            for line in user.splitlines():
                if line.strip().startswith("m.gold") or "-> (m.gold" in line:
                    self.fail("合成拒答样本里仍带 gold 尾路径")

    def test_resampled_answers_come_from_first_block(self):
        extras_c = self.builder.build_synthetic_extras(self.records, self.extras_b)
        for sample in extras_c:
            if sample["_meta"]["is_rejection"]:
                continue
            self.assertEqual(sample["_meta"]["block"], 1)
            self.assertEqual(sample["_meta"]["origin"], "resample")

    def test_synthetic_marked_in_meta(self):
        extras_c = self.builder.build_synthetic_extras(self.records, self.extras_b)
        origins = {s["_meta"]["origin"] for s in extras_c}
        self.assertEqual(origins, {"synthetic_rejection", "resample"})

    def test_build_synthetic_group_keeps_base(self):
        samples = self.builder.build_synthetic_group(self.records, self.extras_b)
        natural = self.builder.build_natural(self.records)
        self.assertEqual(len(samples), len(natural) + len(self.extras_b))


class TestGroupEntryPoint(unittest.TestCase):
    """build_group_samples：三组建集入口的样本构成关系。"""

    def setUp(self):
        pattern = [[3], [25], [], [25]]
        self.records = [make_record(i, 40, pattern[i % 4]) for i in range(12)]
        self.builder = make_builder()

    def sample(self, group):
        return build_group_samples(self.builder, self.records, group)

    def test_natural_only_first_block(self):
        samples = self.sample("natural")
        self.assertEqual(len(samples), 9)
        self.assertTrue(all(s["_meta"]["block"] == 1 for s in samples))

    def test_blocks_keeps_all_first_block_samples(self):
        first = [s for s in self.sample("blocks") if s["_meta"]["block"] == 1]
        natural = self.sample("natural")
        self.assertEqual([s["messages"] for s in first], [s["messages"] for s in natural])

    def test_synthetic_extra_count_matches_blocks(self):
        blocks = self.sample("blocks")
        synthetic = self.sample("synthetic")
        extra_b = len(blocks) - len(self.sample("natural"))
        self.assertEqual(extra_b, 9)
        self.assertEqual(len(synthetic), len(self.sample("natural")) + extra_b)

    def test_synthetic_rejection_share_matches_blocks(self):
        blocks = self.sample("blocks")
        synthetic = self.sample("synthetic")
        extra_b = [s for s in blocks if s["_meta"]["block"] == 2]
        extra_c = [s for s in synthetic if s["_meta"]["origin"] != "block"]
        want = sum(s["_meta"]["is_rejection"] for s in extra_b)
        self.assertEqual(sum(s["_meta"]["is_rejection"] for s in extra_c), want)

    def test_unknown_group_rejected(self):
        with self.assertRaises(ValueError):
            build_group_samples(self.builder, self.records, "nope")

    def test_block_stats(self):
        stats = block_stats(self.sample("blocks"))
        self.assertEqual(stats["n_answer"] + stats["n_rejection"], len(self.sample("blocks")))
        # 12 题里有 3 题 golden 为空（pattern[2]），本来就会被丢弃
        self.assertEqual(stats["by_block"], {1: 9, 2: 9})
        self.assertEqual(set(stats["by_origin"]), {"block"})


class TestRunBuildWithBlocks(unittest.TestCase):
    """CLI 接线：--block_group 走分块构样，产物与 manifest 都要标出组别。"""

    def setUp(self):
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.input_path = self.root / "pool.jsonl"
        pattern = [[3], [25], [], [25]]
        with self.input_path.open("w", encoding="utf-8") as handle:
            for i in range(12):
                handle.write(json.dumps(make_record(i, 40, pattern[i % 4])) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def run_build(self, group, name):
        from kgqa.pfit.build import run_build
        return run_build(dataset="webqsp", input_path=str(self.input_path),
                         exp_dir=str(self.root / name), fmt="v2", path_format="chain",
                         entity_repr="mid", block_group=group, seed=17)

    def read(self, path):
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def test_natural_and_blocks_outputs(self):
        natural = self.read(self.run_build("natural", "a"))
        blocks = self.read(self.run_build("blocks", "b"))
        self.assertEqual(len(natural), 9)
        self.assertEqual(len(blocks), 18)
        self.assertTrue(all("_meta" in s and s["_meta"]["group"] is not None for s in blocks))

    def test_synthetic_output_matches_blocks_extra(self):
        synthetic = self.read(self.run_build("synthetic", "c"))
        self.assertEqual(len(synthetic), 18)
        origins = {s["_meta"]["origin"] for s in synthetic}
        self.assertEqual(origins, {"block", "resample", "synthetic_rejection"})

    def test_manifest_records_block_group(self):
        import json as json_mod
        self.run_build("blocks", "d")
        with (self.root / "d" / "manifest.json").open(encoding="utf-8") as handle:
            manifest = json_mod.load(handle)
        self.assertEqual(manifest["build"]["config"]["block_group"], "blocks")
        self.assertEqual(manifest["build"]["stats"]["by_block"], {"1": 9, "2": 9})

    def test_normal_build_has_no_block_group(self):
        from kgqa.pfit.build import run_build
        out = run_build(dataset="webqsp", input_path=str(self.input_path),
                        exp_dir=str(self.root / "e"), fmt="v2", path_format="chain",
                        entity_repr="mid", seed=17)
        self.assertEqual(len(self.read(out)), 9)


class TestGroupAwareLoader(unittest.TestCase):
    """train 侧接线：keep_group 保留 _group 列，供按组切 train/val。"""

    def setUp(self):
        from tests.kgqa.test_pfit_train_prep import FakeTokenizer
        self.tok = FakeTokenizer()
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sft.jsonl"
        with self.path.open("w", encoding="utf-8") as handle:
            for i in range(5):
                for block in (1, 2):
                    row = {"messages": [{"role": "system", "content": "sys"},
                                        {"role": "user", "content": f"Question: q{i}\n\n1: A -> r -> B"},
                                        {"role": "assistant", "content": "Answer: B"}],
                           "_meta": {"golden_path_indices": [1], "group": i, "block": block}}
                    handle.write(json.dumps(row) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_group_column_present_only_when_requested(self):
        from kgqa.pfit.train import load_dataset_from_jsonl
        with_group = load_dataset_from_jsonl(str(self.path), self.tok, 256, keep_group=True)
        without = load_dataset_from_jsonl(str(self.path), self.tok, 256)
        self.assertIn("_group", with_group.column_names)
        self.assertNotIn("_group", without.column_names)
        self.assertEqual(list(with_group["_group"]), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    def test_group_split_keeps_same_group_together(self):
        from kgqa.pfit.train import load_dataset_from_jsonl
        data = load_dataset_from_jsonl(str(self.path), self.tok, 256, keep_group=True)
        train_idx, val_idx = grouped_split_indices(list(data["_group"]), 0.4, seed=17)
        groups = list(data["_group"])
        self.assertFalse({groups[i] for i in train_idx} & {groups[i] for i in val_idx})


class TestGroupedSplit(unittest.TestCase):
    def test_no_group_crosses_split(self):
        groups = [i // 2 for i in range(100)]
        train_idx, val_idx = grouped_split_indices(groups, 0.2, seed=17)
        self.assertFalse({groups[i] for i in train_idx} & {groups[i] for i in val_idx})

    def test_covers_all_indices_once(self):
        groups = [i // 2 for i in range(100)]
        train_idx, val_idx = grouped_split_indices(groups, 0.2, seed=17)
        self.assertEqual(sorted(train_idx + val_idx), list(range(100)))

    def test_size_close_to_ratio(self):
        groups = [i // 2 for i in range(100)]
        _, val_idx = grouped_split_indices(groups, 0.2, seed=17)
        self.assertLess(abs(len(val_idx) / 100 - 0.2), 0.05)

    def test_deterministic(self):
        groups = [i // 3 for i in range(90)]
        self.assertEqual(grouped_split_indices(groups, 0.2, seed=17),
                         grouped_split_indices(groups, 0.2, seed=17))

    def test_single_group_goes_to_train(self):
        """只有一组时无法切分：整组进训练集，验证集为空（调用方需按空验证集处理）。"""
        train_idx, val_idx = grouped_split_indices([1, 1, 1], 0.2, seed=17)
        self.assertEqual(val_idx, [])
        self.assertEqual(train_idx, [0, 1, 2])

    def test_empty_input(self):
        self.assertEqual(grouped_split_indices([], 0.2, seed=17), ([], []))


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_samples(self):
        records = [make_record(i, 40, [3, 25]) for i in range(5)]
        first = make_builder(seed=17).build_blocks(records)
        second = make_builder(seed=17).build_blocks(records)
        self.assertEqual([s["messages"] for s in first], [s["messages"] for s in second])

    def test_only_second_block_differs_from_first(self):
        """块内打乱只改本块，不改别的块——A 组必须是 B 组的逐条子集。"""
        records = [make_record(i, 40, [3, 25]) for i in range(5)]
        builder = make_builder(seed=17)
        first_blocks = [s for s in builder.build_blocks(records) if s["_meta"]["block"] == 1]
        natural = builder.build_natural(records)
        self.assertEqual([s["messages"] for s in first_blocks], [s["messages"] for s in natural])

    def test_order_independent(self):
        records = [make_record(i, 40, [3, 25]) for i in range(5)]
        shuffled = [records[i] for i in (3, 0, 4, 1, 2)]
        by_key = {(s["_meta"]["group"], s["_meta"]["block"]): s["messages"]
                  for s in make_builder(seed=17).build_blocks(records)}
        for sample in make_builder(seed=17).build_blocks(shuffled):
            key = (sample["_meta"]["group"], sample["_meta"]["block"])
            self.assertEqual(sample["messages"], by_key[key])

    def test_different_seed_shuffles_within_block(self):
        records = [make_record(i, 40, [3, 25]) for i in range(5)]
        first = make_builder(seed=1).build_blocks(records)
        second = make_builder(seed=2).build_blocks(records)
        self.assertNotEqual([s["messages"][1]["content"] for s in first],
                            [s["messages"][1]["content"] for s in second])

    def test_synthetic_group_is_deterministic(self):
        records = [make_record(i, 40, [3, 25]) for i in range(8)]
        extras = make_builder(seed=17).build_synthetic_extras(
            records, make_builder(seed=17).build_blocks(records)[8:])
        again = make_builder(seed=17).build_synthetic_extras(
            records, make_builder(seed=17).build_blocks(records)[8:])
        self.assertEqual([s["messages"] for s in extras], [s["messages"] for s in again])


if __name__ == "__main__":
    unittest.main()
