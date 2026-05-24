import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeTensor:
    def to(self, device):
        return self


class FakeModel:
    def __init__(self):
        self.Msubj = FakeTensor()
        self.Mobj = FakeTensor()
        self.Mrel = FakeTensor()

    def load_state_dict(self, state, strict=False):
        return [], []

    def to(self, device):
        return self


class DumpScoresQaFileTests(unittest.TestCase):
    def test_main_uses_explicit_qa_file_loader_for_non_train_cache(self):
        import WebQSP.dump_scores as dump_scores_mod

        train_loader = Mock()
        cached_test_loader = Mock()
        explicit_loader = Mock()
        fake_model = FakeModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "dump_scores",
                "--input_dir", "data/input/WebQSP",
                "--ckpt", "model.pt",
                "--mode", "test",
                "--qa_file", "QA_data/WebQuestionsSP/qa_test_webqsp_fixed_1581.txt",
                "--output", str(Path(tmpdir) / "cache.pt"),
            ]
            with patch.object(sys, "argv", argv), \
                    patch.object(dump_scores_mod, "load_data", return_value=(
                        {"m.topic": 0}, {"rel": 0}, Mock(), train_loader, cached_test_loader
                    )), \
                    patch.object(dump_scores_mod, "DataLoader", return_value=explicit_loader) as data_loader, \
                    patch.object(dump_scores_mod, "TransferNet", return_value=fake_model), \
                    patch.object(dump_scores_mod.torch, "load", return_value={}), \
                    patch.object(dump_scores_mod.torch.cuda, "is_available", return_value=False), \
                    patch.object(dump_scores_mod, "dump_scores") as dump_scores:
                dump_scores_mod.main()

        data_loader.assert_called_once()
        self.assertIn("qa_test_webqsp_fixed_1581.txt", data_loader.call_args.args[1])
        dump_scores.assert_called_once()
        self.assertIs(dump_scores.call_args.args[1], explicit_loader)
        self.assertEqual(dump_scores.call_args.kwargs["qa_file"], argv[8])


if __name__ == "__main__":
    unittest.main()
