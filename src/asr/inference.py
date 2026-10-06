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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "models/whisper-medium-en-atc-finetuned-full-lora/final"  # merged fp16 model (GPU / conversion source)
DEFAULT_CT2_DIR = DEFAULT_MODEL_DIR.parent / "ct2-int8"  # CTranslate2 int8 copy: all a CPU device needs
SAMPLE_RATE = 16_000
MAX_NEW_TOKENS = 128  # bounds runaway repetition; the longest utterances need ~100 tokens


FULL_WINDOW_S = 30  # Whisper's native input window; the encoder cost scales with the window, not with the clip length


DYNAMIC_WINDOWS = (10, 15, 20, 30)  # seconds; a clip uses the smallest window that fits it. Below 10 s accuracy collapses.


def make_window(model, processor, window_s: int):
    """A (positional-embedding table, feature extractor) pair that makes Whisper accept `window_s` seconds of input.

    The encoder is trained for 1,500 positions (30 s). A shorter window keeps the first `window_s * 50` rows of the
    table and makes the extractor emit spectrograms of that length, which cuts the encoder's work. Measured on this
    project's fine-tuned models, windows of 10 s and up match the 30 s accuracy; 8 s and below do not.
    """
    import torch
    from transformers import WhisperFeatureExtractor

    old = model.model.encoder.embed_positions
    table = torch.nn.Embedding(window_s * 50, old.weight.shape[1])
    table.weight.data = old.weight.data[:window_s * 50].clone().to(old.weight.dtype)
    table.requires_grad_(False)
    fe = processor.feature_extractor
    return table.to(old.weight.device), WhisperFeatureExtractor(
        feature_size=fe.feature_size, sampling_rate=fe.sampling_rate, hop_length=fe.hop_length, chunk_length=window_s,
        n_fft=fe.n_fft, padding_value=fe.padding_value)


def set_window(model, processor, table, extractor) -> None:
    model.model.encoder.embed_positions = table
    model.config.max_source_positions = table.num_embeddings
    processor.feature_extractor = extractor


def pick_window(duration_s: float, windows: tuple[int, ...]) -> int | None:
    """The smallest window (seconds) that fits the clip, or None if it is longer than the largest."""
    return next((w for w in sorted(windows) if duration_s <= w), None)


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class HFTranscriber:
    """Transformers Whisper. `dtype` is fp32, fp16 or bf16; `int8_dynamic` quantizes nn.Linear layers (CPU only)."""

    def __init__(self, model_dir: Path | str, *, device: str = "cpu", dtype: str = "fp32", int8_dynamic: bool = False,
                 threads: int | None = None, max_new_tokens: int = MAX_NEW_TOKENS, windows: tuple[int, ...] = (FULL_WINDOW_S,)):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if threads:
            torch.set_num_threads(threads)
        self._torch, self.device, self.max_new_tokens = torch, device, max_new_tokens
        self.processor = WhisperProcessor.from_pretrained(str(model_dir))
        self.windows = tuple(sorted(windows))
        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        model = WhisperForConditionalGeneration.from_pretrained(str(model_dir), dtype=torch_dtype)
        if int8_dynamic:
            if device != "cpu":
                raise ValueError("int8 dynamic quantization runs on the CPU only")
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            torch_dtype = torch.float32  # activations stay fp32; only the weights are int8
        self.input_dtype = torch_dtype
        self.model = model.to(device).eval()
        # one (positional table, feature extractor) per window; the full 30 s window keeps the model's own
        self._variants = {w: (self.model.model.encoder.embed_positions, self.processor.feature_extractor) if w == FULL_WINDOW_S
                          else make_window(self.model, self.processor, w) for w in self.windows}

    def transcribe(self, audio: np.ndarray) -> str:
        window = pick_window(len(audio) / SAMPLE_RATE, self.windows)
        if window is None:
            raise ValueError(f"clip of {len(audio) / SAMPLE_RATE:.1f}s does not fit the largest window ({self.windows[-1]}s)")
        set_window(self.model, self.processor, *self._variants[window])
        feats = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        with self._torch.inference_mode():
            ids = self.model.generate(feats.to(self.device, dtype=self.input_dtype), max_new_tokens=self.max_new_tokens)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class CT2Transcriber:
    """faster-whisper (CTranslate2) on a converted model directory. `compute_type`: int8, int8_float32, float32, ...

    With the default `windows=(30,)` it uses faster-whisper's own pipeline. With several windows (e.g. DYNAMIC_WINDOWS)
    each clip is encoded in the smallest window that fits it, which makes the encoder (almost all of the CPU time)
    several times cheaper for short clips; clips longer than the largest window fall back to faster-whisper.
    """

    def __init__(self, ct2_dir: Path | str, *, device: str = "cpu", compute_type: str = "int8", threads: int = 0,
                 max_new_tokens: int = MAX_NEW_TOKENS, windows: tuple[int, ...] = (FULL_WINDOW_S,)):
        from faster_whisper import WhisperModel
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_suppressed_tokens

        self.model = WhisperModel(str(ct2_dir), device=device, compute_type=compute_type, cpu_threads=threads)
        self.max_new_tokens, self.windows = max_new_tokens, tuple(sorted(windows))
        self._tokenizer = Tokenizer(self.model.hf_tokenizer, self.model.model.is_multilingual, task="transcribe", language="en")
        self._prompt = self.model.get_prompt(self._tokenizer, [], without_timestamps=True)
        self._suppress = get_suppressed_tokens(self._tokenizer, [-1])

    def _library_transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio.astype(np.float32), language="en", beam_size=1, temperature=0.0, without_timestamps=True,
            condition_on_previous_text=False, vad_filter=False, max_new_tokens=self.max_new_tokens)
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe(self, audio: np.ndarray) -> str:
        window = pick_window(len(audio) / SAMPLE_RATE, self.windows)
        if self.windows == (FULL_WINDOW_S,) or window is None:
            return self._library_transcribe(audio)
        from faster_whisper.audio import pad_or_trim

        feats = self.model.feature_extractor(audio.astype(np.float32))[:, : window * 100]  # 100 mel frames per second
        encoded = self.model.encode(pad_or_trim(feats, length=window * 100))
        out = self.model.model.generate(
            encoded, [self._prompt], beam_size=1, patience=1, length_penalty=1, repetition_penalty=1, no_repeat_ngram_size=0,
            max_length=len(self._prompt) + self.max_new_tokens, suppress_blank=True, suppress_tokens=self._suppress)[0]
        return self._tokenizer.decode(out.sequences_ids[0]).strip()


def convert_to_ct2(model_dir: Path | str, out_dir: Path | str, quantization: str = "int8") -> Path:
    """Convert a Hugging Face Whisper directory to CTranslate2 format (skipped if already converted)."""
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    if (out_dir / "model.bin").exists():
        return out_dir
    import json
    from datetime import datetime, timezone

    import ctranslate2
    from ctranslate2.converters import TransformersConverter

    extra = [f for f in ("tokenizer.json", "preprocessor_config.json") if (model_dir / f).exists()]
    TransformersConverter(str(model_dir), copy_files=extra).convert(str(out_dir), quantization=quantization)
    (out_dir / "conversion.json").write_text(json.dumps({  # provenance: what this directory was made from
        "source_model": model_dir.parent.name, "quantization": quantization, "ctranslate2": ctranslate2.__version__,
        "converted_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    return out_dir


def best_transcriber(model_dir: Path | str | None = None, *, device: str = "auto", threads: int = 4,
                     ct2_dir: Path | str | None = None) -> Transcriber:
    """The fastest backend that keeps the model's accuracy, chosen from the project's benchmarks (res/benchmarks/).

    GPU: PyTorch fp16 with dynamic windows (needs `model_dir`, the merged Hugging Face model). CPU: CTranslate2 int8 with
    dynamic windows, which on the 300-utterance paired check matched the GPU model's WER (-0.25 points, 95% CI -0.70 to
    +0.22) at about 0.7 s per utterance on 4 threads for the medium model (vs ~5 s for PyTorch fp32). On a CPU only the
    CTranslate2 directory is needed (`./download-models.sh --profile cpu`); if it is missing it is converted from
    `model_dir`. Defaults point at the project's medium model. PyTorch's dynamic int8 is deliberately not offered: it
    lost accuracy and needs 3x the memory. Works without torch installed when running on a CPU.
    """
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if device == "cuda":
        return HFTranscriber(model_dir, device="cuda", dtype="fp16", windows=DYNAMIC_WINDOWS)
    ct2 = Path(ct2_dir) if ct2_dir else (model_dir.parent / "ct2-int8" if model_dir != DEFAULT_MODEL_DIR else DEFAULT_CT2_DIR)
    if not (ct2 / "model.bin").exists():
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"no CTranslate2 model in {ct2} and no Hugging Face model in {model_dir} to convert; "
                                    "run ./download-models.sh --profile cpu")
        convert_to_ct2(model_dir, ct2, "int8")
    return CT2Transcriber(ct2, device="cpu", compute_type="int8", threads=threads, windows=DYNAMIC_WINDOWS)
