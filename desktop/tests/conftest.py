import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# the tests report errors through the app log: keep those records out of desktop/logs/bgv.log
os.environ.setdefault("BGV_LOG_DIR", str(Path(tempfile.gettempdir()) / "bgv_test_logs"))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a GPU adapter (offscreen, no window)")
