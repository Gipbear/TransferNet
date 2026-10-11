"""验证输出恢复和缓存写入的原子边界。"""

import json
import tempfile
import unittest
from pathlib import Path

from kgqa.agent.workflow.models import RuntimeQuestion
from kgqa.agent.workflow.run import WorkflowRunner


class TestWorkflowPersistence(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.target = Path(self.temporary.name) / "predictions.jsonl"
        self.question = RuntimeQuestion(0, "q", "Q")
        self.runner = WorkflowRunner.__new__(WorkflowRunner)
        self.row = {"question": self.question.to_dict(),
                    "runtime_sha256": self.runner._question_hash(self.question)}

    def test_torn_last_record_is_backed_up_and_removed(self):
        valid = json.dumps(self.row) + "\n"
        self.target.write_text(valid + '{"question":')
        completed = self.runner._completed(self.target, [self.question])
        self.assertEqual(list(completed), [0])
        self.assertEqual(self.target.read_text(), valid)
        backups = list(self.target.parent.glob("*.torn.*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), '{"question":')

    def test_corrupt_middle_record_is_not_silently_removed(self):
        content = '{"question":\n' + json.dumps(self.row) + "\n"
        self.target.write_text(content)
        with self.assertRaises(ValueError):
            self.runner._completed(self.target, [self.question])
        self.assertEqual(self.target.read_text(), content)

    def test_duplicate_record_is_rejected(self):
        self.target.write_text((json.dumps(self.row) + "\n") * 2)
        with self.assertRaises(ValueError):
            self.runner._completed(self.target, [self.question])
