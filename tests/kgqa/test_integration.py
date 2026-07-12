import os
import tempfile
import unittest
from unittest.mock import patch

from tests.kgqa.integration import artifact_test_available


class TestArtifactGuard(unittest.TestCase):
    def test_disabled_by_default_even_when_artifact_exists(self):
        with tempfile.NamedTemporaryFile() as artifact:
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(artifact_test_available(artifact.name))

    def test_requires_all_artifacts_when_enabled(self):
        with tempfile.NamedTemporaryFile() as artifact:
            with patch.dict(os.environ, {"RUN_KGQA_ARTIFACT_TESTS": "1"}):
                self.assertTrue(artifact_test_available(artifact.name))
                self.assertFalse(artifact_test_available(artifact.name, artifact.name + ".missing"))


if __name__ == "__main__":
    unittest.main()
