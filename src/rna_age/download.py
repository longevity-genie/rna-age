"""Resumable, idempotent downloads into ``data/download/<source>/``.

Every finished file ``X`` gets a sidecar ``X.meta.json`` with the URL, size, checksum and fetch time.
A fetch is skipped when the file and its sidecar agree on size and checksum.

Large files are fetched as parallel HTTP Range segments into ``X.part``; finished segments are recorded in
``X.part.json`` so an interrupted download resumes where it stopped. A ``X.part`` left by another tool
(e.g. ``curl -C -``) without a state file is taken as a finished prefix. Only after the checksum (the
source's own md5 / S3 ETag when it publishes one, else our sha256) and, for ``.gz``, ``gzip -t`` pass is
``X.part`` renamed to ``X``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

MiB = 1024 * 1024
USER_AGENT = "rna-age-downloader/0.1 (+https://github.com/longevity-genie/rna-age)"


@dataclass(frozen=True)
class Remote:
    """A remote file. ``checksum`` is ``"md5:<hex>"``, ``"sha256:<hex>"``, ``"s3etag:<etag>"`` or None."""

    url: str
    name: str
    size: int | None = None
    checksum: str | None = None


def gcs_md5_hex(b64: str) -> str:
    """GCS publishes md5 as base64; convert to hex."""
    return base64.b64decode(b64).hex()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def client(timeout: float = 120.0) -> httpx.Client:
    # httpx honours http(s)_proxy from the environment (trust_env=True by default).
    return httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(timeout, connect=60.0),
        headers={"User-Agent": USER_AGENT},
    )


def head(url: str) -> tuple[int | None, dict[str, str]]:
    with client() as c:
        r = c.head(url)
        if r.status_code >= 400 or "content-length" not in r.headers:
            # Some servers do not answer HEAD properly; ask for one byte instead.
            r = c.get(url, headers={"Range": "bytes=0-0"})
            r.raise_for_status()
            total = r.headers.get("content-range", "").rpartition("/")[2]
            return (int(total) if total.isdigit() else None), dict(r.headers)
        return int(r.headers["content-length"]), dict(r.headers)


# ---------------------------------------------------------------- checksums


def file_digest(path: Path, algo: str, block: int = 16 * MiB) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def s3_etag(path: Path, n_parts: int) -> str:
    """Recompute an S3 multipart ETag ``md5(md5(part1)+md5(part2)+...)-N`` for a guessed part size."""
    size = path.stat().st_size
    for part_mib in (5, 8, 16, 32, 64, 128, 256, 512, 1024):
        part = part_mib * MiB
        if -(-size // part) == n_parts:
            break
    else:
        raise ValueError(f"no standard part size gives {n_parts} parts for {size} bytes")
    digests = []
    with path.open("rb") as f:
        while chunk := f.read(part):
            digests.append(hashlib.md5(chunk).digest())
    return f"{hashlib.md5(b''.join(digests)).hexdigest()}-{n_parts}"


def verify(path: Path, checksum: str | None) -> str:
    """Check ``path`` against ``checksum``; return the checksum string to record (sha256 when none given)."""
    if checksum is None:
        return "sha256:" + file_digest(path, "sha256")
    kind, _, value = checksum.partition(":")
    value = value.strip('"')
    if kind in ("md5", "sha256", "sha1"):
        got = file_digest(path, kind)
    elif kind == "s3etag":
        if "-" in value:
            got = s3_etag(path, int(value.rsplit("-", 1)[1]))
        else:
            got = file_digest(path, "md5")
    else:
        raise ValueError(f"unknown checksum kind {kind!r}")
    if got != value:
        raise ValueError(f"checksum mismatch for {path.name}: expected {kind}:{value}, got {got}")
    return f"{kind}:{value}"


def gzip_ok(path: Path) -> bool:
    return subprocess.run(["gzip", "-t", str(path)], capture_output=True).returncode == 0


# ---------------------------------------------------------------- fetch


def meta_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".meta.json")


def is_fetched(dest: Path, remote: Remote) -> bool:
    mp = meta_path(dest)
    if not (dest.exists() and mp.exists()):
        return False
    meta = json.loads(mp.read_text())
    if dest.stat().st_size != meta["size"]:
        return False
    if remote.size is not None and remote.size != meta["size"]:
        return False
    if remote.checksum is not None and remote.checksum.replace('"', "") != meta["checksum"]:
        return False
    return True


class _Segments:
    """Thread-safe record of finished byte ranges, persisted to ``X.part.json``."""

    def __init__(self, path: Path, size: int, seg: int):
        self.path, self.size, self.seg = path, size, seg
        self.lock = threading.Lock()
        self.done: set[int] = set()
        if path.exists():
            state = json.loads(path.read_text())
            if state["size"] == size and state["segment"] == seg:
                self.done = set(state["done"])

    def todo(self) -> list[int]:
        return [i for i in range(-(-self.size // self.seg)) if i not in self.done]

    def mark(self, i: int) -> None:
        with self.lock:
            self.done.add(i)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"size": self.size, "segment": self.seg, "done": sorted(self.done)}))
            tmp.replace(self.path)


def _fetch_range(url: str, fd: int, start: int, end: int, retries: int = 30) -> None:
    """Fetch bytes [start, end) into ``fd`` at the same offsets, resuming within the range on errors."""
    pos = start
    attempt = 0
    with client() as c:
        while pos < end:
            try:
                with c.stream("GET", url, headers={"Range": f"bytes={pos}-{end - 1}"}) as r:
                    if r.status_code != 206:
                        raise httpx.HTTPStatusError(f"status {r.status_code}", request=r.request, response=r)
                    for chunk in r.iter_bytes(4 * MiB):
                        n = min(len(chunk), end - pos)
                        os.pwrite(fd, chunk[:n], pos)
                        pos += n
                        attempt = 0
            except (httpx.HTTPError, OSError) as e:
                attempt += 1
                if attempt > retries:
                    raise
                wait = min(300, 5 * 2 ** min(attempt, 6))
                log.warning("range %d-%d at %d: %s; retry %d in %ds", start, end, pos, e, attempt, wait)
                time.sleep(wait)


def fetch(
    remote: Remote,
    dest_dir: Path,
    connections: int = 8,
    segment_mib: int = 256,
    polite_delay: float = 0.0,
) -> Path:
    """Download ``remote`` into ``dest_dir`` (idempotent, resumable). Returns the final path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / remote.name
    if is_fetched(dest, remote):
        log.info("skip %s (%s, already fetched and verified)", remote.name, human(dest.stat().st_size))
        return dest

    size = remote.size
    if size is None:
        try:
            size, _ = head(remote.url)
        except httpx.HTTPError:
            size = None
    if size is None:
        return _fetch_stream(remote, dest)

    if dest.exists() and dest.stat().st_size == size:
        # A finished file without a sidecar (or with a stale one): verify instead of re-downloading.
        part = dest
    else:
        part = dest.with_name(dest.name + ".part")
        state = _Segments(dest.with_name(dest.name + ".part.json"), size, segment_mib * MiB)
        if part.exists() and not state.path.exists():
            prefix = part.stat().st_size
            for i in range(prefix // state.seg):
                state.done.add(i)
            log.info("adopting %s of an existing partial download", human(prefix))
        todo = state.todo()
        log.info(
            "fetch %s: %s, %d/%d segments to go, %d connections",
            remote.name,
            human(size),
            len(todo),
            -(-size // state.seg),
            connections,
        )
        fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, size) if os.fstat(fd).st_size < size else None
            t0, done_bytes = time.time(), 0
            with ThreadPoolExecutor(max_workers=max(1, connections)) as pool:
                futures = {}
                for i in todo:
                    start, end = i * state.seg, min(size, (i + 1) * state.seg)
                    futures[pool.submit(_fetch_range, remote.url, fd, start, end)] = (i, end - start)
                    if polite_delay:
                        time.sleep(polite_delay)
                for fut in as_completed(futures):
                    i, n = futures[fut]
                    fut.result()
                    state.mark(i)
                    done_bytes += n
                    rate = done_bytes / max(1e-6, time.time() - t0)
                    left = size - state.seg * len(state.done)
                    log.info(
                        "%s: %d/%d segments, %s/s, ~%.0f min left",
                        remote.name,
                        len(state.done),
                        -(-size // state.seg),
                        human(rate),
                        left / max(rate, 1) / 60,
                    )
            os.fsync(fd)
        finally:
            os.close(fd)

    log.info("verifying %s", remote.name)
    if part.stat().st_size != size:
        raise ValueError(f"{part} has {part.stat().st_size} bytes, expected {size}")
    checksum = verify(part, remote.checksum)
    if remote.name.endswith(".gz") and not gzip_ok(part):
        raise ValueError(f"{part} fails gzip -t")
    if part != dest:
        part.replace(dest)
        dest.with_name(dest.name + ".part.json").unlink(missing_ok=True)
    meta = {
        "url": remote.url,
        "file": dest.name,
        "size": size,
        "checksum": checksum,
        "checksum_source": "publisher" if remote.checksum else "computed",
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    meta_path(dest).write_text(json.dumps(meta, indent=2) + "\n")
    log.info("fetched %s (%s, %s)", dest.name, human(size), checksum)
    return dest


def _fetch_stream(remote: Remote, dest: Path, retries: int = 5) -> Path:
    """Single-stream download for URLs without a known size or Range support (e.g. NCBI GEO dynamic files)."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with client(timeout=300) as c, c.stream("GET", remote.url) as r:
                r.raise_for_status()
                with part.open("wb") as f:
                    for chunk in r.iter_bytes(4 * MiB):
                        f.write(chunk)
            break
        except httpx.HTTPError as e:
            if attempt == retries:
                raise
            wait = 60 * attempt
            log.warning("%s: %s; retry %d in %ds", remote.name, e, attempt, wait)
            time.sleep(wait)
    size = part.stat().st_size
    checksum = verify(part, remote.checksum)
    if remote.name.endswith(".gz") and not gzip_ok(part):
        raise ValueError(f"{part} fails gzip -t")
    part.replace(dest)
    meta = {
        "url": remote.url,
        "file": dest.name,
        "size": size,
        "checksum": checksum,
        "checksum_source": "publisher" if remote.checksum else "computed",
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    meta_path(dest).write_text(json.dumps(meta, indent=2) + "\n")
    log.info("fetched %s (%s, %s)", dest.name, human(size), checksum)
    return dest


def manifest(source_dir: Path) -> list[dict]:
    """All sidecars of a source directory, for the summary."""
    return [json.loads(p.read_text()) for p in sorted(source_dir.rglob("*.meta.json"))]
