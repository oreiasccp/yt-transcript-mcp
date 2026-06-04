"""faster-whisper fallback for caption-less videos. Lazy-loads the model once.

Tuned for an 8GB NVIDIA GPU (RTX 4060 Ti): large-v3 at int8_float16 fits comfortably and
keeps headroom; batched pipeline + VAD cut wall-clock and silence hallucinations.
Override via env: YT_WHISPER_MODEL, YT_WHISPER_DEVICE, YT_WHISPER_COMPUTE.
"""

from __future__ import annotations

import os
from typing import Optional

_MODEL = None
_MODEL_KEY = None

DEFAULT_MODEL = os.environ.get("YT_WHISPER_MODEL", "large-v3")
DEFAULT_DEVICE = os.environ.get("YT_WHISPER_DEVICE", "cuda")
DEFAULT_COMPUTE = os.environ.get("YT_WHISPER_COMPUTE", "int8_float16")
# VAD is OFF by default: the bundled silero VAD over-filters and drops ALL segments on some
# audio with current faster-whisper/onnxruntime. Set YT_WHISPER_VAD=1 to re-enable.
DEFAULT_VAD = os.environ.get("YT_WHISPER_VAD", "0") == "1"


def _load():
    global _MODEL, _MODEL_KEY
    key = (DEFAULT_MODEL, DEFAULT_DEVICE, DEFAULT_COMPUTE)
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL
    from faster_whisper import WhisperModel  # imported lazily so caption-only use needs no CUDA

    try:
        _MODEL = WhisperModel(DEFAULT_MODEL, device=DEFAULT_DEVICE, compute_type=DEFAULT_COMPUTE)
    except Exception:
        # Fall back to CPU int8 if CUDA/cuDNN is unavailable.
        _MODEL = WhisperModel(DEFAULT_MODEL, device="cpu", compute_type="int8")
    _MODEL_KEY = key
    return _MODEL


def transcribe(audio_path: str, language: Optional[str] = None) -> dict:
    """Transcribe a local audio file. Returns {"segments", "lang", "duration"}.

    segments: [{"text", "start", "duration"}] matching the caption-path shape.

    Uses plain (non-batched) decoding with VAD: the BatchedInferencePipeline + VAD combo
    silently drops all segments on some audio in current faster-whisper, so we trade a bit
    of throughput for correctness — fine for one-video-at-a-time local use.
    """
    model = _load()
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=DEFAULT_VAD,
        beam_size=5,
    )
    out = []
    for seg in segments_iter:
        text = seg.text.strip()
        if not text:
            continue
        out.append(
            {
                "text": text,
                "start": round(seg.start, 3),
                "duration": round(seg.end - seg.start, 3),
            }
        )
    return {"segments": out, "lang": info.language, "duration": info.duration}
