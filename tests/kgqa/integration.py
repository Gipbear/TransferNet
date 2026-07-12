"""Guards for tests that depend on gitignored experiment artifacts."""

import os


def artifact_test_available(*paths: str) -> bool:
    """Return true only when artifact tests are enabled and inputs exist."""
    return (
        os.environ.get("RUN_KGQA_ARTIFACT_TESTS") == "1"
        and all(os.path.isfile(path) for path in paths)
    )


ARTIFACT_TEST_SKIP_REASON = (
    "需设置 RUN_KGQA_ARTIFACT_TESTS=1 且本地 ckpt/cache/数据完整"
)
