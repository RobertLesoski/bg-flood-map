"""Storm descriptions for the desktop storm picker / Help tab (bgv/data/storm_blurbs.py)."""
import json

from bgv.data import storm_blurbs as sb


def _root(tmp_path, blurbs=None, info=None):
    (tmp_path / "layers").mkdir()
    if blurbs is not None:
        (tmp_path / "layers" / "storm_blurbs.json").write_text(json.dumps({"storms": blurbs}))
    if info is not None:
        (tmp_path / "layers" / "storm_info.json").write_text(json.dumps({"storms": info}))
    sb.clear_cache()
    return tmp_path


def test_blurb_from_blurbs_file_and_honesty(tmp_path):
    root = _root(tmp_path, {"ev4": {"short": "Historic flood, 2025-04-03", "long": "Long text.", "honesty": ""},
                            "2yr": {"short": "Synthetic 24 h rain", "long": "L", "honesty": "custom line"}})
    b = sb.blurb(root, "ev4")
    assert b["short"].startswith("Historic") and "reconstruction" in b["honesty"] and "not validated" in b["honesty"]
    assert sb.blurb(root, "2yr")["honesty"].startswith("custom line") and "AUC 0.84" in sb.blurb(root, "2yr")["honesty"]
    assert sb.blurb(root, "nope") is None


def test_fallback_to_storm_info(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "_candidates", lambda root: [tmp_path / "layers"])
    root = _root(tmp_path, None, {"5yr": {"title": "5-year design storm", "type": "Design storm", "period": "No date"}})
    b = sb.blurb(root, "5yr")
    assert b and "Design storm" in b["short"] and b["honesty"]


def test_real_blurbs_cover_all_storms_with_event_honesty():
    sb.clear_cache()
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    if not (repo / "layers" / "storm_blurbs.json").is_file():
        import pytest
        pytest.skip("layers/storm_blurbs.json not present")
    for n in ("2yr", "5yr", "10yr", "25yr", "50yr", "100yr", "ev1", "ev2", "ev3", "ev4", "ev5", "ev6"):
        b = sb.blurb(None, n)
        assert b and b["short"], n
        if n.startswith("ev"):
            assert "not a reconstruction" in b["honesty"].lower() or "not validated" in b["honesty"].lower()
