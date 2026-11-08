import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# the tests report errors through the app log: keep those records out of desktop/logs/bgv.log
os.environ.setdefault("BGV_LOG_DIR", str(Path(tempfile.gettempdir()) / "bgv_test_logs"))


import gc
import weakref

import pytest

_LIVE_APPS: "weakref.WeakSet" = weakref.WeakSet()


def _track_apps():
    """Remember every bgv.app.App so a module that forgets app.shutdown() cannot leave its GPU textures behind
    (the full suite ran the adapter out of memory: 'Not enough memory left' in a later module)."""
    try:
        from bgv.app import App
    except Exception:
        return
    if getattr(App, "_test_tracked", False):
        return
    orig = App.__init__

    def __init__(self, *a, **kw):
        _LIVE_APPS.add(self)
        orig(self, *a, **kw)

    App.__init__ = __init__
    App._test_tracked = True


@pytest.fixture(autouse=True, scope="module")
def _shutdown_leaked_apps():
    _track_apps()
    yield
    for app in list(_LIVE_APPS):
        try:
            app.shutdown()
        except Exception:
            pass
    _LIVE_APPS.clear()
    gc.collect()


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a GPU adapter (offscreen, no window)")
    config.addinivalue_line("markers", "real: needs the real data folder (skipped when it is absent)")
