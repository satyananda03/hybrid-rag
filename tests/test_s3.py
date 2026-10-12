from hybrid_rag.config import settings
from hybrid_rag.storage.s3 import objects as s3
from hybrid_rag.storage.s3 import require_bucket, upload_figures_to_s3


class _FakeS3:
    def __init__(self):
        self.puts = []

    def upload_fileobj(self, fh, bucket, key, ExtraArgs):
        if key.endswith("bad.png"):
            raise OSError("boom")
        self.puts.append((bucket, key, ExtraArgs["ContentType"]))


def test_figures_get_urls_and_failures_keep_local_path(tmp_path, monkeypatch):
    fake = _FakeS3()
    monkeypatch.setattr(s3, "get_client", lambda: fake)
    monkeypatch.setattr(settings, "s3_bucket", "bkt")
    monkeypatch.setattr(settings, "s3_region", "ap-southeast-3")
    monkeypatch.setattr(settings, "s3_prefix", "figs/")
    monkeypatch.setattr(s3, "figure_key", lambda p: f"figs/{p.rsplit('/', 1)[-1]}")
    ok, bad = tmp_path / "ok.png", tmp_path / "bad.png"
    ok.write_bytes(b"x"), bad.write_bytes(b"x")
    items = [{"img_path": str(ok)}, {"img_path": str(bad)}]
    upload_figures_to_s3(items)
    assert items[0]["img_path"] == "https://bkt.s3.ap-southeast-3.amazonaws.com/figs/ok.png"
    assert items[1]["img_path"] == str(bad)
    assert fake.puts == [("bkt", "figs/ok.png", "image/png")]


def test_require_bucket_names_the_variable(monkeypatch):
    import pytest

    monkeypatch.setattr(settings, "s3_bucket", None)
    with pytest.raises(ValueError, match="S3_BUCKET"):
        require_bucket()
