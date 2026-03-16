"""
STEP 1 — Download Val Videos ONLY
Run this in a separate terminal alongside step1.py (train)
"""

import json, subprocess, hashlib, time, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR  = r"C:\Users\GPU\Downloads\archive"
OUT_DIR      = r"C:\Users\GPU\Downloads\asl_videos"
FFMPEG_EXE   = r"C:\ffmpeg-2025-12-22-git-c50e5c7778-full_build\bin\ffmpeg.exe"
SPLIT        = "val"
JSON_FILE    = "MSASL_val.json"
MAX_WORKERS  = 4
SKIP_EXISTING = True
MIN_DURATION  = 0.3
MAX_RETRIES   = 3
COOKIE_FILE   = r"C:\Users\GPU\Desktop\cookies.txt"
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("download_val_log.txt", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


def fix_url(url: str) -> str:
    url = url.strip()
    if url.startswith("www."):
        url = "https://" + url
    elif not url.startswith("http"):
        url = "https://www.youtube.com/watch?v=" + url
    return url


def uid_for(entry: dict) -> str:
    key = f"{entry['url']}_{entry['start_time']}_{entry['end_time']}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def download_clip(entry: dict):
    label      = int(entry["label"])
    clean_text = entry.get("clean_text", entry.get("text", str(label)))
    url        = fix_url(entry["url"])
    t_start    = float(entry["start_time"])
    t_end      = float(entry["end_time"])
    duration   = t_end - t_start

    if duration < MIN_DURATION:
        return False, f"SKIP short ({duration:.2f}s) label={label}"

    out_dir  = Path(OUT_DIR) / SPLIT / str(label)
    out_dir.mkdir(parents=True, exist_ok=True)
    uid      = uid_for(entry)
    out_path = out_dir / f"{uid}.mp4"
    tmp_tmpl = out_dir / f"_tmp_{uid}.%(ext)s"

    if SKIP_EXISTING and out_path.exists() and out_path.stat().st_size > 2000:
        return True, f"EXISTS {out_path.name}"

    cmd = [
        "yt-dlp", "--quiet", "--no-warnings",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "--merge-output-format", "mp4",
        "--output", str(tmp_tmpl),
        "--download-sections", f"*{t_start:.3f}-{t_end:.3f}",
        "--force-keyframes-at-cuts",
        "--no-playlist",
    ]
    if COOKIE_FILE and Path(COOKIE_FILE).exists():
        cmd += ["--cookies", COOKIE_FILE]
    cmd.append(url)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[:300].strip())
            written = sorted(out_dir.glob(f"_tmp_{uid}.*"))
            if not written:
                raise FileNotFoundError("No output file produced")
            written[0].rename(out_path)
            return True, f"OK label={label} ({clean_text}) → {out_path.name}"
        except Exception as exc:
            for f in out_dir.glob(f"_tmp_{uid}.*"):
                f.unlink(missing_ok=True)
            if attempt == MAX_RETRIES:
                return False, f"FAIL label={label} ({clean_text}) [{str(exc)[:150]}]"
            time.sleep(2 ** attempt)

    return False, "UNREACHABLE"


if __name__ == "__main__":
    log.info(f"SignBridge — Downloading VAL split")
    log.info(f"Dataset dir : {DATASET_DIR}")
    log.info(f"Output dir  : {OUT_DIR}")

    path = Path(DATASET_DIR) / JSON_FILE
    entries = json.loads(path.read_text(encoding="utf-8"))
    log.info(f"Total val entries: {len(entries)}")

    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_clip, e): e for e in entries}
        for i, fut in enumerate(as_completed(futures), 1):
            success, msg = fut.result()
            if "EXISTS" in msg:
                skip += 1
            elif success:
                ok += 1
                if ok % 50 == 0:
                    log.info(f"[{i}/{len(entries)}] {msg}")
            else:
                fail += 1
                log.warning(f"[{i}/{len(entries)}] {msg}")

            if i % 500 == 0:
                log.info(f"PROGRESS {i}/{len(entries)} OK={ok} FAIL={fail} SKIP={skip}")

    log.info(f"\nVAL done: OK={ok}  FAIL={fail}  SKIP={skip}")