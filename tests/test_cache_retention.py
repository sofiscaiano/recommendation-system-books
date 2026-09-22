from pathlib import Path

from quelibroleo.pipeline import _retain_latest_caches


def test_retention_keeps_latest_evaluate_and_submit(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    kept_eval = cache / "fold_0_eval.joblib"
    kept_submit = cache / "fold_0_submit.joblib"
    stale = cache / "fold_0_stale.joblib"
    for path in (kept_eval, kept_submit, stale):
        path.write_bytes(b"x")
    (cache / "last_submit.txt").write_text("fold_0_submit.joblib\n", encoding="utf-8")

    _retain_latest_caches(cache, "evaluate", {kept_eval})

    assert kept_eval.exists()
    assert kept_submit.exists()
    assert not stale.exists()
    assert (cache / "last_evaluate.txt").read_text(encoding="utf-8") == "fold_0_eval.joblib\n"


def test_retention_waits_for_the_other_manifest(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    current = cache / "fold_0_eval.joblib"
    other = cache / "fold_0_unknown.joblib"
    current.write_bytes(b"x")
    other.write_bytes(b"x")

    _retain_latest_caches(cache, "evaluate", {current})

    assert current.exists()
    assert other.exists()
