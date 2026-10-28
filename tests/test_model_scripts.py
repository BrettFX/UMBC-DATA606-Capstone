"""Tests for the shell helpers behind download-models.sh / upload-models.sh (group and profile selection)."""
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def select(only="", profile=""):
    out = subprocess.run(["bash", "-c", f'source scripts/lib/model_common.sh; select_models "{only}" "{profile}"'], cwd=ROOT,
                         capture_output=True, text=True)
    return out.returncode, [row.split("|")[0] for row in out.stdout.splitlines()], out.stderr


def test_default_selects_every_model():
    assert select()[:2] == (0, ["asr", "asr-ct2", "asr-small-8ep", "asr-results", "ner-data", "ner", "ner-variants"])


def test_cpu_profile_skips_the_large_full_precision_model():
    assert select(profile="cpu")[:2] == (0, ["asr-ct2", "asr-results", "ner-data", "ner"])  # no GPU: the small CTranslate2 model + results + NER
    assert select(profile="gpu")[:2] == (0, ["asr", "asr-results", "ner-data", "ner"])


def test_only_picks_one_group_and_bad_input_is_rejected():
    assert select(only="asr-ct2")[:2] == (0, ["asr-ct2"])
    assert select(only="asr")[1] == ["asr"]  # exactly the group, not every ASR model
    rc, _, err = select(only="tts")
    assert rc != 0 and "groups: asr, asr-ct2, asr-small-8ep, asr-results, ner-data, ner, ner-variants" in err
    rc, _, err = select(only="ner", profile="cpu")
    assert rc != 0 and "not both" in err
    assert select(profile="tpu")[0] != 0


def test_the_ct2_row_uploads_only_its_subfolder():
    out = subprocess.run(["bash", "-c", "source scripts/lib/model_common.sh; printf '%s\\n' \"${SHIP_MODELS[@]}\""], cwd=ROOT, capture_output=True, text=True)
    rows = {r.split("|")[0]: r.split("|") for r in out.stdout.splitlines()}
    assert rows["asr-ct2"][2:5] == ["lora-whisper-medium-en-ct2-int8", "whisper-medium-en-atc-finetuned-full-lora", "ct2-int8"]
    assert rows["asr"][4] == "" and rows["ner"][4] == ""  # the other models upload their usual files


# ---- download-models.sh: S3 -> Google Drive fallback (a stub python3 stands in for AWS and Drive) ----

STUB = """#!/usr/bin/env bash
case "$*" in
  *" probe"*) [ "$STUB_PROBE" = ok ] && exit 0; echo "these AWS credentials cannot read the models bucket"; exit 1 ;;
  *model_store.py*" download "*) [ "$STUB_S3" = ok ] && { echo "S3-OK" >&2; exit 0; }; echo "botocore AccessDenied traceback" >&2; exit 1 ;;
  *app_bundle.py*fetch*) echo "DRIVE-FETCHED"; [ "$STUB_DRIVE" = ok ] && exit 0; echo "drive unreachable" >&2; exit 1 ;;
esac
exit 0
"""


def run_download(tmp_path, *args, probe="fail", s3="fail", drive="ok"):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "python3"
    stub.write_text(STUB)
    stub.chmod(0o755)
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path), "STUB_PROBE": probe, "STUB_S3": s3, "STUB_DRIVE": drive}
    return subprocess.run(["bash", "download-models.sh", "--only", "ner", *args], cwd=ROOT, env=env, capture_output=True, text=True)


def test_auto_falls_back_to_drive_quietly_when_s3_is_not_readable(tmp_path):
    """Credentials for a different AWS account / expired SSO / no awscli: S3 is unusable, so use Drive with no error."""
    r = run_download(tmp_path, probe="fail", drive="ok")
    assert r.returncode == 0
    assert "DRIVE-FETCHED" in r.stdout and "S3 is not available" in r.stdout
    assert "ERROR" not in r.stderr and "AccessDenied" not in r.stderr


def test_auto_falls_back_when_a_download_fails_after_a_good_probe(tmp_path):
    r = run_download(tmp_path, probe="ok", s3="fail", drive="ok")
    assert r.returncode == 0 and "DRIVE-FETCHED" in r.stdout
    assert "AccessDenied" not in r.stderr and "ERROR" not in r.stderr  # the S3 failure is held back, not shown


def test_auto_uses_s3_when_it_works_and_never_touches_drive(tmp_path):
    r = run_download(tmp_path, probe="ok", s3="ok", drive="fail")
    assert r.returncode == 0 and "DRIVE-FETCHED" not in r.stdout and "S3-OK" in r.stderr


def test_auto_errors_only_when_every_method_fails(tmp_path):
    r = run_download(tmp_path, probe="ok", s3="fail", drive="fail")
    assert r.returncode == 1
    assert "all download methods failed" in r.stderr and "AccessDenied" in r.stderr  # now the S3 detail is shown
    r = run_download(tmp_path, probe="fail", drive="fail")
    assert r.returncode == 1 and "could not download the models from S3 or Google Drive" in r.stderr


def test_explicit_s3_never_falls_back(tmp_path):
    r = run_download(tmp_path, "--source", "s3", s3="fail", drive="ok")
    assert r.returncode == 1 and "DRIVE-FETCHED" not in r.stdout and "AccessDenied" in r.stderr


def test_explicit_gdrive_skips_s3(tmp_path):
    r = run_download(tmp_path, "--source", "gdrive", probe="ok", s3="ok", drive="ok")
    assert r.returncode == 0 and "DRIVE-FETCHED" in r.stdout and "S3-OK" not in r.stderr


# ---- artifacts.check_access: why S3 is (not) usable ----

def _err(name, code=""):
    e = type(name, (Exception,), {})("boom")
    e.response = {"Error": {"Code": code}}
    return e


class _Client:
    def __init__(self, exc=None, count=1):
        self.exc, self.count = exc, count

    def list_objects_v2(self, **kw):
        if self.exc:
            raise self.exc
        return {"KeyCount": self.count}


def test_check_access_reports_why_s3_is_unusable():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import artifacts
    base = "s3://bucket/ml-tasks"
    assert artifacts.check_access(_Client(), base) is None
    assert "different AWS account" in artifacts.check_access(_Client(_err("ClientError", "AccessDenied")), base)
    assert "no AWS credentials" in artifacts.check_access(_Client(_err("NoCredentialsError")), base)
    assert "expired" in artifacts.check_access(_Client(_err("TokenRetrievalError")), base)
    assert "expired" in artifacts.check_access(_Client(_err("ClientError", "ExpiredToken")), base)
    assert "cannot reach" in artifacts.check_access(_Client(_err("EndpointConnectionError")), base)
    assert "no models found" in artifacts.check_access(_Client(count=0), base)
    assert artifacts.check_access(_Client(), "not-an-s3-uri")  # a bad URI is reported, never raised
