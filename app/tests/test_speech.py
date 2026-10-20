import io
import wave

import numpy as np
import pytest

from app.services import speech
from app.services.errors import AppError


class FakeKokoro:
    def __call__(self, text, voice, speed):
        self.seen = (text, voice, speed)
        yield "gs", "ps", np.zeros(2400, dtype=np.float32)
        yield "gs", "ps", np.full(2400, 0.5, dtype=np.float32)


def test_prepare_text_collapses_whitespace():
    assert speech.prepare_text("  descend   flight\nlevel one ") == "descend flight level one"


def test_synthesize_concatenates_chunks_into_24k_mono_wav():
    fake = FakeKokoro()
    wav = speech.synthesize("descend  flight level one zero zero", voice="af_heart", speed=1.1, pipeline=fake)
    assert fake.seen == ("descend flight level one zero zero", "af_heart", 1.1)
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (24000, 1, 4800)


def test_empty_transcript_is_an_app_error():
    with pytest.raises(AppError):
        speech.synthesize("   ", pipeline=FakeKokoro())
