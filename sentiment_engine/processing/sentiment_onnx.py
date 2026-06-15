"""Optional ONNX-int8 sentiment scorer (CryptoBERT/FinBERT, no PyTorch).

This is the lightweight path for resource-constrained servers: a transformer
exported to ONNX and int8-quantized runs in `onnxruntime` with a `tokenizers`
tokenizer — roughly 300-400 MB resident, versus ~1.5-2 GB for the PyTorch build.
No torch is imported at runtime.

Off by default. Produce the model with convert_cryptobert_onnx.py (run once on a
machine that has torch), copy the output to the server, then set
SENTIMENT_ONNX_MODEL / SENTIMENT_ONNX_TOKENIZER. If onnxruntime/tokenizers aren't
installed or the model can't load, scoring falls back to the lexicon.

The logits->score mapping is a pure function, unit-testable without the heavy deps.
"""

from __future__ import annotations

import logging
import math
from decimal import Decimal
from typing import List, Optional, Sequence, Tuple

from sentiment_engine.config import ONNX_LABELS, ONNX_MODEL_PATH, ONNX_TOKENIZER_PATH
from sentiment_engine.processing.sentiment_transformer import score_from_distribution

logger = logging.getLogger(__name__)

_session = None
_tokenizer = None
_load_failed = False
_MAX_LEN = 128


def onnx_enabled() -> bool:
    return bool(ONNX_MODEL_PATH and ONNX_TOKENIZER_PATH)


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def logits_to_score(logits: Sequence[float], labels: Sequence[str]) -> Tuple[float, float]:
    """Class logits + label order -> (sentiment in [-1,1], confidence). Pure."""
    probs = _softmax(list(logits))
    dist = [{"label": (labels[i] if i < len(labels) else f"label_{i}"), "score": probs[i]}
            for i in range(len(probs))]
    return score_from_distribution(dist)


def _load() -> bool:
    global _session, _tokenizer, _load_failed
    if _load_failed:
        return False
    if _session is not None and _tokenizer is not None:
        return True
    try:
        import onnxruntime as ort  # heavy import, only when enabled
        from tokenizers import Tokenizer
        _session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
        _tokenizer = Tokenizer.from_file(ONNX_TOKENIZER_PATH)
        try:
            _tokenizer.enable_truncation(max_length=_MAX_LEN)
        except Exception:  # noqa: BLE001 - truncation config is best-effort
            pass
        logger.info("onnx scorer: loaded %s", ONNX_MODEL_PATH)
        return True
    except Exception as exc:  # noqa: BLE001 - any failure -> fall back to lexicon
        _load_failed = True
        logger.warning("onnx scorer: could not load (%s); using lexicon", type(exc).__name__)
        return False


def score_with_onnx(text: str) -> Optional[Tuple[Decimal, Decimal]]:
    """(sentiment, confidence) Decimals, or None if the ONNX model is unavailable."""
    if not _load():
        return None
    try:
        import numpy as np
        enc = _tokenizer.encode(text)
        ids = list(enc.ids)[:_MAX_LEN]
        attn = [1] * len(ids)
        inputs = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([attn], dtype=np.int64),
        }
        input_names = {i.name for i in _session.get_inputs()}
        if "token_type_ids" in input_names:
            inputs["token_type_ids"] = np.array([[0] * len(ids)], dtype=np.int64)
        outputs = _session.run(None, {k: v for k, v in inputs.items() if k in input_names})
        logits = list(outputs[0][0])
        labels = [lab.strip().lower() for lab in ONNX_LABELS.split(",") if lab.strip()]
        sentiment, confidence = logits_to_score(logits, labels)
        return Decimal(str(round(sentiment, 4))), Decimal(str(round(confidence, 4)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("onnx scorer: inference failed (%s); using lexicon", type(exc).__name__)
        return None
