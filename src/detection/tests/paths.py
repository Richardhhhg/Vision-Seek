from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TESTS_DIR = Path(__file__).resolve().parent

REPO_ROOT = str(_REPO_ROOT)
TESTS_DIR = str(_TESTS_DIR)
TEST_VIDEO_PATH = str(_REPO_ROOT / "data" / "test_data" / "test_video.mp4")
TEST_MODEL_PATH = str(_REPO_ROOT / "src" / "detection" / "detection_model" / "model" / "test_model.pt")
TEST_CONFIG_REAL_PATH = str(_TESTS_DIR / "test_config_real.json")
TEST_CONFIG_FAKE_PATH = str(_TESTS_DIR / "test_config_fake.json")
FAKE_MODEL_PATH = str(_REPO_ROOT / "fake_path" / "model.pt")
