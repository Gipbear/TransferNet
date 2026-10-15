"""测试多实体交叉连接关系提取与 SPARQL 生成逻辑。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "data/analysis/20261007_1749__cwq_full_refusal_audit"
if str(AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIT_DIR))

from multi_entity_verifier import build_sparql_query  # noqa: E402


class TestMultiEntityVerification(unittest.TestCase):
    def test_sparql_builder_1hop_fwd(self):
        cands = ["m.01jt2w", "m.02d413"]
        sec = "m.03cq17y"
        pat = ("1hop_fwd", "education.educational_institution.newspaper")
        query = build_sparql_query(cands, sec, pat)
        self.assertIn("VALUES ?x { ns:m.01jt2w ns:m.02d413 }", query)
        self.assertIn("?x ns:education.educational_institution.newspaper ns:m.03cq17y .", query)

    def test_sparql_builder_2hop_fwd_fwd(self):
        cands = ["m.064qvcw", "m.02vynz1"]
        sec = "g.121sxjvg"
        pat = ("2hop_fwd_fwd", "film.film.starring", "film.performance.character")
        query = build_sparql_query(cands, sec, pat)
        self.assertIn("VALUES ?x { ns:m.064qvcw ns:m.02vynz1 }", query)
        self.assertIn("?x ns:film.film.starring ?mid .", query)
        self.assertIn("?mid ns:film.performance.character ns:g.121sxjvg .", query)

    def test_qlever_execution_sample_201(self):
        from multi_entity_verifier import (
            build_sparql_query,
            execute_qlever_tsv,
        )

        cands = ["m.064qvcw", "m.02vynz1", "m.010tvsqx"]
        sec = "g.121sxjvg"
        pat = ("2hop_fwd_fwd", "film.film.starring", "film.performance.character")
        sparql = build_sparql_query(cands, sec, pat)
        matched = execute_qlever_tsv(sparql)
        self.assertEqual(matched, ["m.064qvcw"])


if __name__ == "__main__":
    unittest.main()
