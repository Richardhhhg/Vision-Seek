from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TESTS_DIR = Path(__file__).resolve().parent

REPO_ROOT = str(_REPO_ROOT)
TESTS_DIR = str(_TESTS_DIR)
TEST_VIDEO_PATH = str(_REPO_ROOT / "data" / "test_data" / "test_5.mp4")
TEST_SRT_PATH = str(_REPO_ROOT / "data" / "test_data" / "test_5.srt")
TEST_TELEMETRY_PATH = str(_REPO_ROOT / "data" / "test_data" / "test_5_telemetry.csv")
