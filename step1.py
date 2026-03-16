"""
STEP 1 — Download Train Videos ONLY
Resumes from where it left off (SKIP_EXISTING = True)
Fixes PermissionError on temp file cleanup
"""

import json, subprocess, hashlib, time, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── HARDCODED FFMPEG PATH ─────────────────────────────────────────────────────
FFMPEG_EXE = r"C:\ffmpeg-2025-12-22-git-c50e5c7778-full_build\bin\ffmpeg.exe"
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR   = r"C:\Users\GPU\Downloads\archive"
OUT_DIR       = r"C:\Users\GPU\Downloads\asl_videos"
SPLIT         = "train"
JSON_FILE     = "MSASL_train.json"
MAX_WORKERS   = 4
SKIP_EXISTING = True    # resumes — skips already downloaded files
MIN_DURATION  = 0.3
MAX_RETRIES   = 2       # reduced — unavailable videos waste time
COOKIE_FILE   = r"C:\Users\GPU\Desktop\cookies.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("download_train_log.txt", encoding="utf-8"),
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


def cleanup_tmp(out_dir, uid):
    """Safely delete temp files — ignores PermissionError if file still in use."""
    for f in out_dir.glob(f"_tmp_{uid}.*"):
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass  # file still locked by yt-dlp — will be cleaned next run


def download_clip(entry: dict):
    label      = int(entry["label"])
    clean_text = entry.get("clean_text", entry.get("text", str(label)))
    raw_url    = entry["url"]
    url        = fix_url(raw_url)
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

    # skip if already downloaded successfully
    if SKIP_EXISTING and out_path.exists() and out_path.stat().st_size > 2000:
        return True, f"EXISTS {out_path.name}"

    # skip if previously confirmed unavailable (saves retry time)
    fail_marker = out_dir / f"_fail_{uid}.txt"
    if fail_marker.exists():
        return False, f"SKIP known-fail label={label} ({clean_text})"

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
                err = r.stderr[:300].strip()
                # permanently mark unavailable videos so we skip them next run
                if "Video unavailable" in err or "Private video" in err:
                    try:
                        fail_marker.write_text("unavailable", encoding="utf-8")
                    except Exception:
                        pass
                    return False, f"UNAVAILABLE label={label} ({clean_text})"
                raise RuntimeError(err)

            written = sorted(out_dir.glob(f"_tmp_{uid}.*"))
            if not written:
                raise FileNotFoundError("No output file produced")

            written[0].rename(out_path)
            return True, f"OK label={label} ({clean_text}) dur={duration:.2f}s → {out_path.name}"

        except subprocess.TimeoutExpired:
            msg = "timeout"
        except Exception as exc:
            msg = str(exc)[:200]

        cleanup_tmp(out_dir, uid)

        if attempt == MAX_RETRIES:
            return False, f"FAIL label={label} ({clean_text}) [{msg}]"
        time.sleep(2 ** attempt)

    return False, "UNREACHABLE"


def verify_setup():
    ok = True
    try:
        subprocess.run([FFMPEG_EXE, "-version"],
                       capture_output=True, check=True, timeout=10)
        log.info(f"ffmpeg : ✓  ({FFMPEG_EXE})")
    except Exception as e:
        log.error(f"ffmpeg NOT FOUND: {e}")
        ok = False
    try:
        subprocess.run(["yt-dlp", "--version"],
                       capture_output=True, check=True, timeout=10)
        log.info("yt-dlp : ✓")
    except Exception:
        log.error("yt-dlp NOT FOUND — run: pip install yt-dlp")
        ok = False
    if COOKIE_FILE and Path(COOKIE_FILE).exists():
        log.info(f"cookies: ✓  ({COOKIE_FILE})")
    else:
        log.warning(f"cookies: NOT FOUND at {COOKIE_FILE} — some videos may fail")
    return ok


def save_class_map():
    cp = Path(DATASET_DIR) / "MSASL_classes.json"
    if not cp.exists():
        return
    classes = json.loads(cp.read_text(encoding="utf-8"))
    mapping = {str(i): w for i, w in enumerate(classes)}
    out = Path(OUT_DIR) / "class_map.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    log.info(f"class_map.json saved ({len(mapping)} classes)")


if __name__ == "__main__":
    log.info("SignBridge — Step 1: Download TRAIN Videos")
    log.info(f"Dataset dir  : {DATASET_DIR}")
    log.info(f"Output dir   : {OUT_DIR}")
    log.info(f"Workers      : {MAX_WORKERS}")
    log.info(f"Cookie file  : {COOKIE_FILE}")
    log.info(f"Skip existing: {SKIP_EXISTING}")

    if not verify_setup():
        log.error("Fix the above issues then re-run.")
        exit(1)

    save_class_map()

    path    = Path(DATASET_DIR) / JSON_FILE
    entries = json.loads(path.read_text(encoding="utf-8"))

    # count what's already on disk
    already = sum(
        1 for e in entries
        if (Path(OUT_DIR) / SPLIT / str(int(e["label"])) /
            f"{uid_for(e)}.mp4").exists()
    )

    log.info(f"\n{'='*60}")
    log.info(f"Split: TRAIN  |  {len(entries)} total entries")
    log.info(f"Already on disk  : {already}  (will be skipped instantly)")
    log.info(f"Remaining to try : {len(entries) - already}")
    log.info(f"{'='*60}\n")

    ok = fail = skip = unavail = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_clip, e): e for e in entries}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                success, msg = fut.result()
            except Exception as exc:
                fail += 1
                log.warning(f"  [{i}/{len(entries)}] EXCEPTION: {exc}")
                continue

            if "EXISTS" in msg:
                skip += 1
            elif "UNAVAILABLE" in msg or "known-fail" in msg:
                unavail += 1
            elif success:
                ok += 1
                log.info(f"  [{i}/{len(entries)}] {msg}")
            else:
                fail += 1
                log.warning(f"  [{i}/{len(entries)}] {msg}")

            if i % 500 == 0:
                log.info(
                    f"  PROGRESS {i}/{len(entries)} "
                    f"NEW={ok} FAIL={fail} SKIP={skip} UNAVAIL={unavail}"
                )

    log.info(f"\nTRAIN complete:")
    log.info(f"  New downloads : {ok}")
    log.info(f"  Unavailable   : {unavail}  (deleted/private YouTube videos)")
    log.info(f"  Other fails   : {fail}")
    log.info(f"  Skipped       : {skip}  (already existed)")
    log.info(f"\nNext: run step2.py to extract keypoints")