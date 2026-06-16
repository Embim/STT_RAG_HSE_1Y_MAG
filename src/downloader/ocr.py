import logging
import re
import threading
import cv2
import easyocr
import torch

logger = logging.getLogger(__name__)

_reader = None
_reader_lock = threading.Lock()
_OCR_MIN_CONFIDENCE = 0.3
_OCR_MAX_NOISE_RATIO = 0.3
_OCR_MIN_ALPHA_NUM_CHARS = 4

def get_reader():
    global _reader
    if _reader is None:
        with _reader_lock:
            if _reader is None:
                use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
                logger.info(f"Initializing EasyOCR reader (ru, en), gpu={use_gpu}")
                _reader = easyocr.Reader(['ru', 'en'], gpu=use_gpu)
    return _reader


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_noise(text: str) -> bool:
    cleaned = _normalize_text(text)
    if not cleaned:
        return True
    alnum_count = sum(1 for ch in cleaned if ch.isalnum())
    if alnum_count < _OCR_MIN_ALPHA_NUM_CHARS:
        return True
    noise_count = sum(1 for ch in cleaned if not ch.isalnum() and not ch.isspace())
    noise_ratio = noise_count / max(1, len(cleaned))
    return noise_ratio > _OCR_MAX_NOISE_RATIO


def _is_near_duplicate(current: str, previous: str) -> bool:
    if not previous:
        return False
    cur = _normalize_text(current).lower()
    prev = _normalize_text(previous).lower()
    if not cur or not prev:
        return False
    if cur == prev:
        return True
    if cur in prev or prev in cur:
        return True
    cur_words = set(cur.split())
    prev_words = set(prev.split())
    if not cur_words or not prev_words:
        return False
    overlap = len(cur_words & prev_words) / max(1, min(len(cur_words), len(prev_words)))
    return overlap >= 0.85


def extract_text_from_video(video_path: str, interval_sec: int = 15) -> list[dict]:
    """
    Extracts text from a video by processing one frame every `interval_sec` seconds.
    Returns a list of segment dicts: [{"start": t, "end": t + interval_sec, "text": "[ВИЗУАЛЬНЫЙ ТЕКСТ НА ЭКРАНЕ: ...]"}, ...]
    """
    logger.info("Starting OCR extraction for %s (interval: %ds)", video_path, interval_sec)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Failed to open video %s for OCR", video_path)
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
        
    frame_interval = max(1, int(fps * interval_sec))
    reader = get_reader()

    segments = []
    last_kept_text = ""
    
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time_sec = frame_idx / fps
        
        logger.debug("Processing OCR for frame at %.1fs", current_time_sec)
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Mild denoise + adaptive threshold often helps on slides/screens.
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            prepared = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
            )

            raw_results = reader.readtext(prepared, detail=1, paragraph=False)
            kept_lines = []
            for result in raw_results:
                if len(result) < 3:
                    continue
                _, candidate_text, confidence = result
                candidate_text = _normalize_text(str(candidate_text))
                if not candidate_text:
                    continue
                if float(confidence or 0.0) < _OCR_MIN_CONFIDENCE:
                    continue
                if _looks_like_noise(candidate_text):
                    continue
                kept_lines.append(candidate_text)

            text = _normalize_text(" ".join(kept_lines))
            
            if text:
                if _is_near_duplicate(text, last_kept_text):
                    frame_idx += frame_interval
                    continue
                segments.append({
                    "start": current_time_sec,
                    "end": current_time_sec + interval_sec,
                    "text": f"\n[ВИЗУАЛЬНЫЙ ТЕКСТ НА ЭКРАНЕ: {text}]\n"
                })
                last_kept_text = text
        except Exception as e:
            logger.error("OCR failed at %.1fs: %s", current_time_sec, e)
            
        frame_idx += frame_interval
        
    cap.release()
    logger.info("OCR extraction finished, found %d text segments", len(segments))
    return segments

def merge_ocr_segments(asr_segments: list[dict], ocr_segments: list[dict]) -> list[dict]:
    """
    Merges OCR segments into ASR segments and sorts them by start time.
    """
    merged = list(asr_segments) + list(ocr_segments)
    merged.sort(key=lambda x: float(x.get("start", 0.0) or 0.0))
    return merged
