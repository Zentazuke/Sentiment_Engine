"""Convert CryptoBERT (or any HF text-classification model) to an int8 ONNX model.

Run this ONCE on a machine that has PyTorch (your laptop / a beefy box) — NOT on
the 4 GB server. It exports the model to ONNX and int8-quantizes it, producing a
~110 MB model plus its tokenizer. Then copy the output folder to the server,
which only needs `onnxruntime` + `tokenizers` (see requirements-onnx.txt) — no
PyTorch at all.

    pip install "optimum[onnxruntime]" transformers torch
    python convert_cryptobert_onnx.py                 # default: ElKulako/cryptobert
    python convert_cryptobert_onnx.py ProsusAI/finbert ./finbert_onnx

After it finishes it prints the exact env vars to set on the server, e.g.:
    set SENTIMENT_ONNX_MODEL=.../cryptobert_onnx/model_quantized.onnx
    set SENTIMENT_ONNX_TOKENIZER=.../cryptobert_onnx/tokenizer.json
    set SENTIMENT_ONNX_LABELS=bearish,neutral,bullish     (match the model's id2label order)
"""

import sys
from pathlib import Path


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "ElKulako/cryptobert"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "./cryptobert_onnx").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoConfig, AutoTokenizer

    print(f"Exporting {model_id} to ONNX in {out_dir} ...")
    ORTModelForSequenceClassification.from_pretrained(model_id, export=True).save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)

    print("Quantizing to int8 (dynamic) ...")
    quantizer = ORTQuantizer.from_pretrained(out_dir)
    qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=out_dir, quantization_config=qconfig)

    # Report the label order so SENTIMENT_ONNX_LABELS is correct.
    cfg = AutoConfig.from_pretrained(model_id)
    id2label = getattr(cfg, "id2label", {}) or {}
    labels = ",".join(str(id2label[i]).lower() for i in sorted(id2label)) if id2label else "bearish,neutral,bullish"

    quantized = next((p for p in out_dir.glob("*quantized*.onnx")), None)
    model_path = quantized or next(out_dir.glob("*.onnx"))
    print("\nDone. On the server set:")
    print(f"  SENTIMENT_ONNX_MODEL={model_path}")
    print(f"  SENTIMENT_ONNX_TOKENIZER={out_dir / 'tokenizer.json'}")
    print(f"  SENTIMENT_ONNX_LABELS={labels}")


if __name__ == "__main__":
    main()
