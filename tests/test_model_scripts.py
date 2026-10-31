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


# ---- download-models.sh: Google Drive by default, the S3 experiment store only when asked (a stub python3 stands in for both) ----

STUB = """#!/usr/bin/env bash
echo "CALL $*" >> "$STUB_LOG"
case "$*" in
  *model_store.py*" download "*) [ "$STUB_S3" = ok ] && { echo "S3-OK" >&2; exit 0; }; echo "botocore AccessDenied traceback" >&2; exit 1 ;;
  *app_bundle.py*fetch*) echo "DRIVE-FETCHED $*"; [ "$STUB_DRIVE" = ok ] && exit 0; echo "drive unreachable" >&2; exit 1 ;;
esac
exit 0
"""


def run_download(tmp_path, *args, s3="ok", drive="ok", store=None):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "python3"
    stub.write_text(STUB)
    stub.chmod(0o755)
    log = tmp_path / "calls.log"
    log.write_text("")
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "HOME": str(tmp_path), "STUB_LOG": str(log), "STUB_S3": s3, "STUB_DRIVE": drive}
    if store:
        env["MODEL_STORE_BASE"] = store
    r = subprocess.run(["bash", "download-models.sh", *args], cwd=ROOT, env=env, capture_output=True, text=True)
    return r, log.read_text()


def test_default_downloads_the_production_models_from_drive_without_touching_s3(tmp_path):
    r, calls = run_download(tmp_path, s3="fail")
    assert r.returncode == 0 and "DRIVE-FETCHED" in r.stdout and "model_store.py" not in calls
    assert "--force" not in calls


def test_force_and_dry_run_on_drive(tmp_path):
    r, calls = run_download(tmp_path, "--force")
    assert r.returncode == 0 and "fetch --force" in calls
    r, calls = run_download(tmp_path, "--dry-run")
    assert r.returncode == 0 and "Would download clearanceiq-models.zip" in r.stdout and "app_bundle.py" not in calls


def test_a_drive_failure_is_an_error_and_does_not_fall_back_to_s3(tmp_path):
    r, calls = run_download(tmp_path, drive="fail", store="s3://bucket/ml-tasks")
    assert r.returncode == 1 and "drive unreachable" in r.stderr and "model_store.py" not in calls


def test_the_cpu_profile_is_what_the_drive_bundle_already_is(tmp_path):
    for args in (["--profile", "cpu"], ["--only", "ner"], ["--only", "asr-ct2"]):
        r, calls = run_download(tmp_path, *args)
        assert r.returncode == 0 and "app_bundle.py" in calls, args


def test_options_that_need_the_s3_store_are_refused_without_source_s3(tmp_path):
    for args in (["--profile", "gpu"], ["--only", "asr"], ["--only", "asr-results"], ["--run-id", "x"]):
        r, calls = run_download(tmp_path, *args)
        assert r.returncode == 2 and "--source s3" in r.stderr and calls == "", args


def test_s3_needs_a_configured_store_and_never_falls_back(tmp_path):
    r, calls = run_download(tmp_path, "--source", "s3")
    assert r.returncode == 2 and "MODEL_STORE_BASE" in r.stderr and calls == ""  # nothing hard-coded to fall back on
    r, calls = run_download(tmp_path, "--source", "s3", "--only", "ner", store="s3://bucket/ml-tasks", s3="fail")
    assert r.returncode == 1 and "AccessDenied" in r.stderr and "app_bundle.py" not in calls
    r, calls = run_download(tmp_path, "--source", "s3", "--only", "ner", store="s3://bucket/ml-tasks")
    assert r.returncode == 0 and "--base s3://bucket/ml-tasks" in calls and "app_bundle.py" not in calls


def test_the_scripts_parse_and_the_s3_scripts_name_no_bucket():
    for f in ("setup.sh", "download-models.sh", "upload-models.sh"):
        assert subprocess.run(["bash", "-n", f], cwd=ROOT).returncode == 0, f
    import re

    for f in ("src/artifacts.py", "scripts/model_store.py", "scripts/lib/model_common.sh", "upload-models.sh", "download-models.sh", "README.md", "docs/report.md"):
        named = re.findall(r"s3://(?!<bucket>|bucket|my-bucket|other-bucket)[A-Za-z0-9][A-Za-z0-9._-]*", (ROOT / f).read_text())
        assert not named, (f, named)  # the private bucket is configured through MODEL_STORE_BASE, never written into the repository


# ---- setup.sh: Jupyter kernel registration must survive a broken pyzmq/ipykernel (a stub conda records the calls) ----

CONDA_STUB = """#!/usr/bin/env bash
echo "$*" >> "$STUB_LOG"
case "$*" in
  *"import zmq"*) [ -e "$STUB_FIXED" ] || [ "$STUB_BROKEN" != 1 ] ;;
  *"pip uninstall"*) exit 0 ;;
  install*) [ "$STUB_REPAIR" = ok ] && touch "$STUB_FIXED"; [ "$STUB_REPAIR" = ok ] ;;
  *"ipykernel install"*) [ "$STUB_KERNEL" = ok ] && echo "Installed kernelspec" ;;
esac
"""


def register(tmp_path, broken=False, repair="ok", kernel="ok"):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    (bindir / "conda").write_text(CONDA_STUB)
    (bindir / "conda").chmod(0o755)
    log = tmp_path / "calls.log"
    log.write_text("")
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "STUB_LOG": str(log), "STUB_FIXED": str(tmp_path / "fixed"), "STUB_BROKEN": "1" if broken else "0",
           "STUB_REPAIR": repair, "STUB_KERNEL": kernel}
    r = subprocess.run(["bash", "-c", "source scripts/lib/jupyter_env.sh; register_jupyter_kernel data-science"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    return r, log.read_text()


def test_kernel_registration_is_plain_when_the_env_is_healthy(tmp_path):
    r, calls = register(tmp_path)
    assert r.returncode == 0 and "pip uninstall" not in calls and "install --user" in calls


def test_a_broken_pyzmq_is_repaired_then_the_kernel_is_registered(tmp_path):
    """Re-running setup.sh left pip's and conda's pyzmq/ipykernel mixed: `import zmq` fails until they are reinstalled cleanly."""
    r, calls = register(tmp_path, broken=True)
    assert r.returncode == 0 and "Repairing" in r.stdout
    order = [calls.index(s) for s in ("pip uninstall", "install -y -q -n data-science", "ipykernel install")]
    assert order == sorted(order)  # uninstall pip's copy, reinstall via conda, then register


def test_an_unrepairable_env_only_warns(tmp_path):
    r, calls = register(tmp_path, broken=True, repair="fail")
    assert r.returncode == 1 and "WARNING: could not register" in r.stderr and "ipykernel install" not in calls


def test_a_failing_kernel_install_only_warns(tmp_path):
    r, _ = register(tmp_path, kernel="fail")
    assert r.returncode == 1 and "WARNING: could not register" in r.stderr
