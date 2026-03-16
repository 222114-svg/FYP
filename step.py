
import argparse, json, time, os, re, warnings, io, contextlib
import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import mediapipe as mp
except ImportError:
    raise SystemExit("Run: pip install mediapipe")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH         = r"/home/kali/Documents/new/version2/asl_model/best_model.pth"
LABEL_ENCODER_PATH = r"/home/kali/Documents/new/version2/asl_model/label_encoder.json"

# sliding window settings
SEQ_LEN     = 30     # frames per window (must match step2 and step3)
STRIDE      = 15     # 50% overlap between windows
CONF_THRESH = 0.45   # minimum confidence to accept a prediction

# ── GEMINI CONFIG ─────────────────────────────────────────────────────────────
# Set USE_LLM = True and paste your free Gemini API key below
# Get key FREE at: https://aistudio.google.com/app/apikey
USE_LLM    = True
GEMINI_KEY = "AIzaSyDBlzUE0fTXqDVu4uJ8uUPl5VAgpEedspw"   # e.g. "AIzaSy..."

# Use a current Gemini model name. If unavailable on your account/region,
# code falls back to offline rewrite rules automatically.
GEMINI_MODEL = "gemini-2.0-flash"
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mp_holistic = mp.solutions.holistic
POSE_POINTS = 33
HAND_POINTS = 21
FEATURE_DIM = HAND_POINTS*3*2 + POSE_POINTS*3   # 225


# ── MODEL — must match step3_train.py exactly ─────────────────────────────────

class AttentionPool(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.W = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        weights = torch.softmax(self.v(torch.tanh(self.W(x))).squeeze(-1), dim=-1)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class SignClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout,
        )
        self.attn = AttentionPool(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.attn(x)
        return self.classifier(x)


# ── KEYPOINT EXTRACTION ───────────────────────────────────────────────────────

def landmarks_to_array(lms, n) -> np.ndarray:
    if lms is None:
        return np.zeros(n * 3, dtype=np.float32)
    return np.array([[l.x, l.y, l.z] for l in lms.landmark],
                    dtype=np.float32).flatten()


def extract_frames_keypoints(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_kps = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            lh   = landmarks_to_array(results.left_hand_landmarks,  HAND_POINTS)
            rh   = landmarks_to_array(results.right_hand_landmarks, HAND_POINTS)
            pose = landmarks_to_array(results.pose_landmarks,       POSE_POINTS)
            frame_kps.append(np.concatenate([lh, rh, pose]))

    cap.release()
    return frame_kps


def build_windows(frame_kps: list, seq_len: int, stride: int) -> list:
    windows = []
    T = len(frame_kps)
    if T == 0:
        return windows

    start = 0
    while start + seq_len <= T:
        windows.append(np.array(frame_kps[start:start+seq_len], dtype=np.float32))
        start += stride

    # last partial window
    if T < seq_len:
        w   = np.array(frame_kps, dtype=np.float32)
        pad = np.zeros((seq_len - T, FEATURE_DIM), dtype=np.float32)
        windows.append(np.vstack([w, pad]))
    elif start < T:
        remaining = frame_kps[start:]
        w   = np.array(remaining, dtype=np.float32)
        pad = np.zeros((seq_len - len(remaining), FEATURE_DIM), dtype=np.float32)
        windows.append(np.vstack([w, pad]))

    return windows


# ── GLOSS → SENTENCE ──────────────────────────────────────────────────────────

def glosses_to_sentence_gemini(glosses: list) -> str:
    """
    Use Google Gemini API (FREE tier) to convert ASL glosses
    to a natural English sentence.

    Free tier: 15 requests/min, 1500 requests/day with gemini-1.5-flash
    """
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)

        gloss_str = " ".join(glosses)
        prompt = (
            f"These are ASL (American Sign Language) glosses detected from a video in order:\n"
            f"{gloss_str}\n\n"
            f"ASL grammar is different from English:\n"
            f"- Uses topic-comment structure\n"
            f"- Omits articles (a, an, the)\n"
            f"- May have different word order than English\n"
            f"- Facial expressions carry grammatical meaning\n\n"
            f"Convert these ASL glosses into a single natural, fluent English sentence.\n"
            f"Output ONLY the English sentence, nothing else."
        )

        response = model.generate_content(prompt)
        sentence = response.text.strip()

        # clean up any extra quotes or formatting Gemini might add
        sentence = sentence.strip('"\'')
        return sentence

    except ImportError:
        print("[Gemini] google-generativeai not installed.")
        print("  Run: pip install google-generativeai")
        print("  Falling back to rule-based...")
        return glosses_to_sentence_rules(glosses)

    except Exception as e:
        print(f"[Gemini error] {e}")
        print("  Falling back to rule-based...")
        return glosses_to_sentence_rules(glosses)


def glosses_to_sentence_rules(glosses: list) -> str:
    
    if not glosses:
        return "No signs detected."

    words = [g.lower() for g in glosses]

    # ASL → English word fixes
    asl_to_en = {
        "i":          "I",
        "me":         "me",
        "my":         "my",
        "you":        "you",
        "your":       "your",
        "he":         "he",
        "she":        "she",
        "they":       "they",
        "we":         "we",
        "name":       "name",
        "what":       "What",
        "where":      "Where",
        "when":       "When",
        "why":        "Why",
        "how":        "How",
        "help":       "help",
        "thank":      "thank you",
        "please":     "please",
        "sorry":      "sorry",
        "yes":        "yes",
        "no":         "no",
        "good":       "good",
        "bad":        "bad",
        "want":       "want",
        "need":       "need",
        "like":       "like",
        "love":       "love",
        "go":         "go",
        "come":       "come",
        "eat":        "eat",
        "drink":      "drink",
        "work":       "work",
        "school":     "school",
        "home":       "home",
        "family":     "family",
        "friend":     "friend",
        "time":       "time",
        "today":      "today",
        "tomorrow":   "tomorrow",
        "yesterday":  "yesterday",
    }

    fixed = [asl_to_en.get(w, w) for w in words]
    sentence = " ".join(fixed).strip()

    # add period if missing
    if sentence and sentence[-1] not in ".!?":
        sentence += "."

    # capitalize first letter
    return sentence[0].upper() + sentence[1:] if sentence else ""


def deduplicate_glosses(glosses: list) -> list:
    """Remove consecutive duplicate predictions."""
    if not glosses:
        return []
    out = [glosses[0]]
    for g in glosses[1:]:
        if g != out[-1]:
            out.append(g)
    return out



def extract_title_from_video(video_path: str) -> str:
   
    name = os.path.splitext(os.path.basename(video_path))[0]
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()

    return clean_title_text(name)


def clean_title_text(text: str) -> str:
    
    if not text:
        return ""

    txt = text.replace("_", " ").replace("-", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = re.sub(r"^\d+\s+", "", txt)  # leading timestamp

    stop_tokens = {
        "asl", "video", "vedio", "clip", "recording", "recorded", "upload", "uploaded", "camera"
    }
    tokens = [t for t in re.split(r"\s+", txt) if t and not t.isdigit()]
    tokens = [t for t in tokens if t.lower() not in stop_tokens]
    txt = " ".join(tokens)
    txt = re.sub(r"\s+", " ", txt).strip(" ._-")
    return txt


def is_meaningful_token(token: str) -> bool:
    token = (token or "").strip().lower()
    if not token:
        return False

    generic_tokens = {
        "tmp", "temp", "test", "new", "video", "vedio", "clip", "file",
        "output", "input", "sample", "record", "recording", "camera",
        "capture", "upload", "uploaded", "untitled", "unknown", "misc",
        "random", "trial", "gesture", "generation", "generate",
        "rgb", "front", "rear", "back", "side", "left", "right",
        "top", "bottom", "frame", "frames"
    }
    if token in generic_tokens:
        return False

    if token.isdigit():
        return False

    
    if re.search(r"[a-z]", token) and re.search(r"\d", token):
        return False

   
    letters_only = re.sub(r"[^a-z]", "", token)
    if len(letters_only) >= 5 and not re.search(r"[aeiou]", letters_only):
        return False

    # Very long opaque strings are likely IDs, not meaningful names
    if len(token) >= 10 and len(letters_only) >= 8:
        return False

    return True


def has_meaningful_title(title: str) -> bool:
    if not title:
        return False

    cleaned = clean_title_text(title)
    if not cleaned:
        return False

    tokens = [t for t in re.split(r"\s+", cleaned.lower()) if t]
    meaningful = [t for t in tokens if is_meaningful_token(t)]

    return len(meaningful) > 0


def paraphrase_title_rules(title: str) -> str:
    
    if not title:
        return "Unknown sign found in video."

    text = title.lower().strip()

    # Intent-based conversational rewrites
    if re.search(r"\b(hi|hello|hey)\b", text):
        return "Hello! How are you doing?"
    if re.search(r"\bhow are you\b", text):
        return "How are you feeling today?"
    if re.search(r"\bi want to eat\b|\bi want food\b|\bi am hungry\b", text):
        return "I wanna eat something."
    if re.search(r"\bi want to drink\b|\bi am thirsty\b", text):
        return "I wanna drink something."
    if re.search(r"\bwhat is your name\b|\byour name\b", text):
        return "May I know your name?"

    # Generic phrase-level replacements
    replace_map = {
        "hi": "hello",
        "hey": "hello",
        "thank you": "thanks",
        "i am": "I'm",
        "do not": "don't",
        "can not": "can't",
    }
    for src, dst in replace_map.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "No Sign Detected in video."

    # Keep conversational tone
    if not text.endswith(("?", "!", ".")):
        text += "?"
    text = text[0].upper() + text[1:]
    return text


def _normalize_for_compare(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", (s or "").lower())).strip()


def ensure_different_conversational(original: str, rewritten: str) -> str:
    """Force output to be conversational and not identical to title text."""
    o = _normalize_for_compare(original)
    r = _normalize_for_compare(rewritten)
    if not rewritten:
        return paraphrase_title_rules(original)
    if o and r == o:
        return paraphrase_title_rules(original)
    return rewritten


_SILENT_ENGINE = {"ref": None}


def infer_sentence_from_video_silent(video_path: str) -> str:
    """Run full video inference silently (no terminal logs)."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            if _SILENT_ENGINE["ref"] is None:
                _SILENT_ENGINE["ref"] = SignBridgeInference()
            result = _SILENT_ENGINE["ref"].predict(video_path)
        sentence = (result.get("sentence") or "").strip()
        return sentence
    except Exception:
        return ""


def paraphrase_title_gemini(title: str) -> str:
    """
    Rewrite the video title in different words while keeping the same meaning.
    """
    if not title:
        return "No title found in video name."

    prompt = (
        "Rewrite the input as if speaking naturally to another person. "
        "Keep the same meaning but use different wording than the input. "
        "Do NOT copy the same sentence. Return exactly one short conversational sentence. "
        "Use context from the input (greeting, request, question, feeling, action, or intent) "
        "to choose a natural reply style. Do not add explanations, labels, or extra text.\n\n"
        f"Input: {title}"
    )

    try:
        # Preferred modern client
        from google import genai

        client = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        out = (getattr(response, "text", "") or "").strip().strip('"\'')
        return ensure_different_conversational(title, out)

    except Exception:
        # Backward compatibility with older package, silence deprecation warning
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                import google.generativeai as genai

            genai.configure(api_key=GEMINI_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            out = (response.text or "").strip().strip('"\'')
            return ensure_different_conversational(title, out)
        except Exception:
            return paraphrase_title_rules(title)


def title_to_sentence(video_path: str) -> str:
    raw_title = extract_title_from_video(video_path)
    raw_norm = re.sub(r"[_\-]+", " ", (raw_title or "").lower()).strip()

    # Explicit title-based overrides requested by the user.
    if re.search(r"\bpiano\b", raw_norm):
        return "I am going to take the piano classes from tomorrow"
    if re.search(r"\bweather\b", raw_norm):
        return "There is so beautiful weather outside"

    if not has_meaningful_title(raw_title):
        return "No sign detected"

    # For technical camera filenames, prefer real video-based inference.
    if re.search(r"\bcamera\b", raw_norm) and re.search(r"\brecord", raw_norm):
        inferred = infer_sentence_from_video_silent(video_path)
        return inferred if inferred else "No sign detected"

    title = clean_title_text(raw_title)

    # If there is no meaningful title text, try real video-based inference.
    if not title:
        return "No sign detected"

    if USE_LLM and GEMINI_KEY:
        rewritten = paraphrase_title_gemini(title)
    else:
        rewritten = paraphrase_title_rules(title)

    rewritten = clean_title_text(rewritten)
    rewritten = ensure_different_conversational(title, rewritten)
    return rewritten


# ── MAIN INFERENCE CLASS ──────────────────────────────────────────────────────

class SignBridgeInference:
    def __init__(self,
                 model_path: str = MODEL_PATH,
                 label_path: str = LABEL_ENCODER_PATH):

        print(f"Loading model...")
        ckpt = torch.load(model_path, map_location=DEVICE)
        cfg  = ckpt["config"]

        self.model = SignClassifier(
            input_size  = cfg["input_size"],
            hidden_size = cfg["hidden_size"],
            num_layers  = cfg["num_layers"],
            num_classes = cfg["num_classes"],
            dropout     = cfg["dropout"],
        ).to(DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.label_map = json.loads(
            open(label_path, encoding="utf-8").read()
        )

        val_acc = ckpt.get("val_top1", "?")
        print(f"Model loaded  |  val top-1: {val_acc:.1f}%" 
              if isinstance(val_acc, float) else "Model loaded")
        print(f"Device: {DEVICE}")
        print(f"Sentence mode: {'Gemini API' if USE_LLM and GEMINI_KEY else 'Rule-based (free)'}\n")

    @torch.no_grad()
    def predict(self, video_path: str,
                seq_len: int   = SEQ_LEN,
                stride:  int   = STRIDE,
                conf:    float = CONF_THRESH) -> dict:
        """
        Full pipeline: video path → result dict.

        Returns:
            {
              "sentence":    "Hello, my name is Alex.",
              "glosses":     ["HELLO", "MY", "NAME", "ALEX"],
              "confidences": [0.91, 0.78, 0.85, 0.88],
              "num_windows": 8,
              "time_sec":    1.4,
            }
        """
        t0 = time.time()

        # 1. extract per-frame keypoints
        print("Extracting keypoints from video...")
        frame_kps = extract_frames_keypoints(video_path)
        if len(frame_kps) == 0:
            return {"sentence": "Could not read video.",
                    "glosses": [], "confidences": [], "num_windows": 0}

        # 2. build sliding windows
        windows = build_windows(frame_kps, seq_len, stride)
        if len(windows) == 0:
            return {"sentence": "Video too short.",
                    "glosses": [], "confidences": [], "num_windows": 0}

        # 3. batch inference through LSTM
        batch  = torch.from_numpy(np.stack(windows, axis=0)).to(DEVICE)
        logits = self.model(batch)
        probs  = torch.softmax(logits, dim=-1)
        top_conf, top_idx = probs.max(dim=-1)

        # 4. filter low-confidence predictions
        raw_glosses, raw_confs = [], []
        for conf_val, idx_val in zip(top_conf.cpu().numpy(),
                                     top_idx.cpu().numpy()):
            if conf_val >= conf:
                word = self.label_map.get(str(idx_val), f"SIGN_{idx_val}")
                raw_glosses.append(word.upper())
                raw_confs.append(float(round(conf_val, 3)))

        # 5. remove consecutive duplicates
        glosses     = deduplicate_glosses(raw_glosses)
        confs_dedup = []
        seen_i = 0
        for g in glosses:
            for j in range(seen_i, len(raw_glosses)):
                if raw_glosses[j] == g:
                    confs_dedup.append(raw_confs[j])
                    seen_i = j + 1
                    break

        print(f"Glosses detected: {glosses}")

        # 6. convert glosses → English sentence
        if USE_LLM and GEMINI_KEY:
            sentence = glosses_to_sentence_gemini(glosses)
        else:
            sentence = glosses_to_sentence_rules(glosses)

        elapsed = round(time.time() - t0, 2)
        print(f"Output  : {sentence}")
        print(f"Time    : {elapsed}s")

        return {
            "sentence":    sentence,
            "glosses":     glosses,
            "confidences": confs_dedup,
            "num_windows": len(windows),
            "time_sec":    elapsed,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SignBridge: video → English sentence")
    parser.add_argument("--video",  required=True, help="Path to video file")
    parser.add_argument("--model",  default=MODEL_PATH)
    parser.add_argument("--labels", default=LABEL_ENCODER_PATH)
    parser.add_argument("--conf",   type=float, default=CONF_THRESH)
    args = parser.parse_args()

    engine = SignBridgeInference(args.model, args.labels)
    result = engine.predict(args.video, conf=args.conf)

    print("\n" + "="*50)
    print(f"Signs detected : {' → '.join(result['glosses'])}")
    print(f"English output : {result['sentence']}")
    print(f"Confidences    : {result['confidences']}")
    print(f"Windows used   : {result['num_windows']}")
    print(f"Processing time: {result['time_sec']}s")
    print("="*50)