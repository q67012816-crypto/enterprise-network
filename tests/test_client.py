import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import dart_client as dc
import pytest

FAKE = "A" * 40


def make(tmp_path, responses, **kw):
    calls = []
    def fetcher(url):
        calls.append(url); return responses.pop(0)
    c = dc.DartClient(raw_dir=tmp_path / "raw", state_dir=tmp_path / "state",
                      min_interval=0, fetcher=fetcher, sleep=lambda s: None, **kw)
    return c, calls


def test_redact(monkeypatch):
    monkeypatch.setenv("OPENDART_API_KEY", FAKE)
    s = dc.redact(f"https://x/api/a.json?crtfc_key={FAKE}&corp_code=1 and {FAKE}")
    assert FAKE not in s and "crtfc_key=***" in s


def test_cache_and_no_key_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDART_API_KEY", FAKE)
    ok = json.dumps({"status": "000", "list": []}).encode()
    c, calls = make(tmp_path, [ok])
    p = {"corp_code": "00126380", "bsns_year": "2025", "reprt_code": "11011"}
    assert c.get_json("otrCprInvstmntSttus.json", p)["status"] == "000"
    assert c.get_json("otrCprInvstmntSttus.json", p)["status"] == "000"  # cache hit
    assert len(calls) == 1 and c.calls_today() == 1
    for f in (tmp_path).rglob("*"):
        if f.is_file():
            assert FAKE not in f.read_text(errors="ignore")


def test_retry_then_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDART_API_KEY", FAKE)
    c, calls = make(tmp_path, [b'{"status":"800"}', b'{"status":"020"}'])
    with pytest.raises(dc.DailyLimitReached):
        c.get("company.json", {"corp_code": "1"})
    assert len(calls) == 2


def test_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDART_API_KEY", FAKE)
    c, _ = make(tmp_path, [b'{"status":"000"}'], daily_budget=1)
    c.get("company.json", {"corp_code": "1"})
    with pytest.raises(dc.DailyLimitReached):
        c.get("company.json", {"corp_code": "2"})


def test_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENDART_API_KEY", raising=False)
    c, _ = make(tmp_path, [])
    with pytest.raises(RuntimeError, match="not set"):
        c.get("company.json", {"corp_code": "1"})
