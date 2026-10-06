"""Speech-to-text backends behind one interface, shared by the app and the latency benchmarks.

Every backend turns a mono 16 kHz float waveform into text with greedy decoding:
- `HFTranscriber`: transformers Whisper on CPU or GPU (fp32 / fp16 / bf16), optionally with int8 dynamic
  quantization of the linear layers (CPU only).
- `CT2Transcriber`: CTranslate2 (faster-whisper) with int8 weights, usually the fastest option on CPU. It runs
  a converted copy of the model; `convert_to_ct2` creates it from a Hugging Face model directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

SAMPLE_RATE = 16_000
MAX_NEW_TOKENS = 128  # bounds runaway repetition; the longest utterances need ~100 tokens


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class HFTranscriber:
    """Transformers Whisper. `dtype` is fp32, fp16 or bf16; `int8_dynamic` quantizes nn.Linear layers (CPU only)."""

    def __init__(self, model_dir: Path | str, *, device: str = "cpu", dtype: str = "fp32", int8_dynamic: bool = False,
                 threads: int | None = None, max_new_tokens: int = MAX_NEW_TOKENS):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if threads:
            torch.set_num_threads(threads)
        self._torch, self.device, self.max_new_tokens = torch, device, max_new_tokens
        self.processor = WhisperProcessor.from_pretrained(str(model_dir))
        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        model = WhisperForConditionalGeneration.from_pretrained(str(model_dir), dtype=torch_dtype)
        if int8_dynamic:
            if device != "cpu":
                raise ValueError("int8 dynamic quantization runs on the CPU only")
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            torch_dtype = torch.float32  # activations stay fp32; only the weights are int8
        self.input_dtype = torch_dtype
        self.model = model.to(device).eval()

    def transcribe(self, audio: np.ndarray) -> str:
        feats = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        with self._torch.inference_mode():
            ids = self.model.generate(feats.to(self.device, dtype=self.input_dtype), max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class CT2Transcriber:
    """faster-whisper (CTranslate2) on a converted model directory. `compute_type`: int8, int8_float32, float32, ..."""

    def __init__(self, ct2_dir: Path | str, *, device: str = "cpu", compute_type: str = "int8", threads: int = 0,
                 max_new_tokens: int = MAX_NEW_TOKENS):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(str(ct2_dir), device=device, compute_type=compute_type, cpu_threads=threads)
        self.max_new_tokens = max_new_tokens

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio.astype(np.float32), language="en", beam_size=1, temperature=0.0, without_timestamps=True,
            condition_on_previous_text=False, vad_filter=False, max_new_tokens=self.max_new_tokens)
        return " ".join(s.text.strip() for s in segments).strip()


def convert_to_ct2(model_dir: Path | str, out_dir: Path | str, quantization: str = "int8") -> Path:
    """Convert a Hugging Face Whisper directory to CTranslate2 format (skipped if already converted)."""
    from ctranslate2.converters import TransformersConverter

    model_dir, out_dir = Path(model_dir), Path(out_dir)
    if (out_dir / "model.bin").exists():
        return out_dir
    extra = [f for f in ("tokenizer.json", "preprocessor_config.json") if (model_dir / f).exists()]
    TransformersConverter(str(model_dir), copy_files=extra).convert(str(out_dir), quantization=quantization)
    return out_dir
