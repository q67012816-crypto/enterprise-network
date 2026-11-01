"""Resumable OpenDART client.

- API key read ONLY from env var OPENDART_API_KEY (never stored/logged).
- Raw responses cached under data/raw/<endpoint>/<sha1(params)>.<ext>; cache hits cost no call.
- Daily call counter persisted in data/state/calls.json (KST date) so runs can resume.
- Rate limiting + retry with exponential backoff; status 020 (limit exceeded) stops the run.
"""
from __future__ import annotations
import hashlib, json, os, re, time, datetime as dt
from pathlib import Path
import urllib.request, urllib.parse, urllib.error, threading

BASE = "https://opendart.fss.or.kr/api/"
ROOT = Path(__file__).resolve().parents[1]
KST = dt.timezone(dt.timedelta(hours=9))
# Official guide: status 020 "일반적으로는 20,000건 이상의 요청에 대하여" -> we stay well below.
DEFAULT_DAILY_BUDGET = int(os.environ.get("DART_DAILY_BUDGET", "18000"))


class DailyLimitReached(RuntimeError):
    pass


def redact(text: str) -> str:
    """Remove crtfc_key values (and the raw key if present in env) from any string."""
    text = re.sub(r"(crtfc_key=)[^&\s\"']+", r"\1***", str(text))
    key = os.environ.get("OPENDART_API_KEY")
    if key:
        text = text.replace(key, "***")
    return text


def cache_key(endpoint: str, params: dict) -> str:
    p = {k: v for k, v in sorted(params.items()) if k != "crtfc_key"}
    return hashlib.sha1(json.dumps([endpoint, p], ensure_ascii=False).encode()).hexdigest()


class DartClient:
    def __init__(self, raw_dir: Path | None = None, state_dir: Path | None = None,
                 min_interval: float = float(os.environ.get("DART_MIN_INTERVAL", "0.25")), max_retries: int = 4,
                 daily_budget: int = DEFAULT_DAILY_BUDGET, fetcher=None, sleep=time.sleep):
        self.raw_dir = Path(raw_dir or ROOT / "data/raw")
        self.state_dir = Path(state_dir or ROOT / "data/state")
        self.min_interval, self.max_retries, self.daily_budget = min_interval, max_retries, daily_budget
        self.fetcher = fetcher or self._http_get
        self.sleep = sleep
        self._last = 0.0
        self._lock = threading.Lock()

    # ---- state ----
    def _state_path(self):
        return self.state_dir / "calls.json"

    def _load_state(self):
        today = dt.datetime.now(KST).strftime("%Y-%m-%d")
        try:
            s = json.loads(self._state_path().read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            s = {}
        if s.get("date") != today:
            s = {"date": today, "calls": 0}
        return s

    def calls_today(self) -> int:
        return self._load_state()["calls"]

    def _bump(self):
        import fcntl  # cross-process lock so concurrent scripts don't overwrite the counter
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_dir / "calls.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            s = self._load_state(); s["calls"] += 1
            tmp = self._state_path().with_suffix(".tmp"); tmp.write_text(json.dumps(s)); tmp.replace(self._state_path())

    # ---- http ----
    @staticmethod
    def _http_get(url: str) -> bytes:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()

    def _key(self) -> str:
        key = os.environ.get("OPENDART_API_KEY")
        if not key:
            raise RuntimeError("OPENDART_API_KEY is not set (export it in your shell; never commit it)")
        return key

    def get(self, endpoint: str, params: dict | None = None, ext: str | None = None, force=False):
        """Return raw bytes for endpoint (e.g. 'otrCprInvstmntSttus.json'). Cached."""
        params = dict(params or {})
        ext = ext or endpoint.rsplit(".", 1)[-1]
        path = self.raw_dir / endpoint.rsplit(".", 1)[0] / f"{cache_key(endpoint, params)}.{ext}"
        if path.exists() and not force:
            return path.read_bytes()
        if self.calls_today() >= self.daily_budget:
            raise DailyLimitReached(f"daily budget {self.daily_budget} reached; rerun tomorrow to resume")
        url = BASE + endpoint + "?" + urllib.parse.urlencode({"crtfc_key": self._key(), **params})
        err = None
        for attempt in range(self.max_retries + 1):
            with self._lock:  # global spacing + counter, thread-safe
                if self.calls_today() >= self.daily_budget:
                    raise DailyLimitReached(f"daily budget {self.daily_budget} reached")
                wait = self.min_interval - (time.monotonic() - self._last)
                if wait > 0:
                    self.sleep(wait)
                self._last = time.monotonic()
                self._bump()
            try:
                body = self.fetcher(url)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                err = redact(repr(e)); self.sleep(2 ** attempt); continue
            status = _status(body, ext)
            if status == "020":
                raise DailyLimitReached("OpenDART status 020 (request limit exceeded)")
            if status in ("800", "900"):
                err = f"status {status}"; self.sleep(2 ** attempt); continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(
                {"endpoint": endpoint, "params": params, "status": status,
                 "fetched_at": dt.datetime.now(KST).isoformat()}, ensure_ascii=False))
            path.write_bytes(body)  # cache 000/013 etc.; 013 = no data, is informative
            return body
        raise RuntimeError(redact(f"{endpoint} failed after retries: {err}"))

    def get_json(self, endpoint: str, params: dict | None = None) -> dict:
        return json.loads(self.get(endpoint, params).decode("utf-8"))


def _status(body: bytes, ext: str) -> str | None:
    if ext == "json":
        try:
            return json.loads(body.decode("utf-8")).get("status")
        except Exception:
            return None
    if body[:2] == b"PK":  # zip (corpCode.xml)
        return "000"
    m = re.search(rb"<status>(\d+)</status>", body[:500])
    return m.group(1).decode() if m else None
