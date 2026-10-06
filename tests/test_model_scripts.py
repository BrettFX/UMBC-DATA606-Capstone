"""Tests for the shell helpers behind download-models.sh / upload-models.sh (group and profile selection)."""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def select(only="", profile=""):
    out = subprocess.run(["bash", "-c", f'source scripts/lib/model_common.sh; select_models "{only}" "{profile}"'], cwd=ROOT,
                         capture_output=True, text=True)
    return out.returncode, [row.split("|")[0] for row in out.stdout.splitlines()], out.stderr


def test_default_selects_every_model():
    assert select()[:2] == (0, ["asr", "asr-ct2", "ner"])


def test_cpu_profile_skips_the_large_full_precision_model():
    assert select(profile="cpu")[:2] == (0, ["asr-ct2", "ner"])  # no GPU: the small CTranslate2 model + NER
    assert select(profile="gpu")[:2] == (0, ["asr", "ner"])


def test_only_picks_one_group_and_bad_input_is_rejected():
    assert select(only="asr-ct2")[:2] == (0, ["asr-ct2"])
    assert select(only="asr")[1] == ["asr"]  # exactly the group, not every ASR model
    rc, _, err = select(only="tts")
    assert rc != 0 and "groups: asr, asr-ct2, ner" in err
    rc, _, err = select(only="ner", profile="cpu")
    assert rc != 0 and "not both" in err
    assert select(profile="tpu")[0] != 0


def test_the_ct2_row_uploads_only_its_subfolder():
    out = subprocess.run(["bash", "-c", "source scripts/lib/model_common.sh; printf '%s\\n' \"${SHIP_MODELS[@]}\""], cwd=ROOT, capture_output=True, text=True)
    rows = {r.split("|")[0]: r.split("|") for r in out.stdout.splitlines()}
    assert rows["asr-ct2"][2:5] == ["lora-whisper-medium-en-ct2-int8", "whisper-medium-en-atc-finetuned-full-lora", "ct2-int8"]
    assert rows["asr"][4] == "" and rows["ner"][4] == ""  # the other models upload their usual files
