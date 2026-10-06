"""Tests for benchmark aggregation, history logging, baseline comparison and the report (no models needed)."""
import csv
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import benchmarking as b  # noqa: E402

INFO = {"cpu": "TestCPU", "logical_cores": 8, "ram_gb": 16.0, "gpu": None, "wsl2": False, "python": "3.12", "git_commit": "abc123",
        "git_dirty": False, "versions": {"torch": "2.5", "ctranslate2": "4.8"}}


def result(median, wer=0.05, **kw):
    return {"n": 10, "median_s": median, "p95_s": median * 1.3, "mean_s": median, "rtf": 0.5, "wer": wer, "cer": 0.02, "load_s": 3.0,
            "warmup_s": 4.0, "peak_rss_mb": 2000.0, **kw}


def test_aggregate_computes_latency_rtf_and_wer():
    out = b.aggregate([1.0, 2.0, 3.0, 10.0], [2.0, 2.0, 2.0, 2.0], ["contact rhein one three two", "roger", "climb", "descend"],
                      ["contact rhein one three two", "roger", "climb", "descend now"])
    assert out["n"] == 4 and out["median_s"] == 2.5 and out["max_s"] == 10.0
    assert out["rtf"] == pytest.approx(16.0 / 8.0)  # total processing time over total audio time
    assert out["wer"] == pytest.approx(1 / 9)  # one word missing out of nine reference words


def test_history_appends_rows_and_baseline_is_the_latest_baseline_run(tmp_path):
    hist = tmp_path / "history.csv"
    b.append_history(hist, "r1", "t1", "baseline", INFO, {"a/ct2/cpu-4t": result(3.0), "ner/spacy/cpu": result(0.001)})
    b.append_history(hist, "r2", "t2", "experiment", INFO, {"a/ct2/cpu-4t": result(1.5)})
    b.append_history(hist, "r3", "t3", "baseline", INFO, {"a/ct2/cpu-4t": result(2.8)})  # a re-measured baseline replaces r1
    rows = list(csv.DictReader(open(hist)))
    assert [r["run_id"] for r in rows] == ["r1", "r1", "r2", "r3"] and rows[0]["git_commit"] == "abc123"
    base = b.baseline_rows(hist)
    assert set(base) == {"a/ct2/cpu-4t"} and float(base["a/ct2/cpu-4t"]["median_s"]) == 2.8
    assert b.baseline_rows(tmp_path / "missing.csv") == {}


def test_report_shows_speedup_and_changes_against_baseline():
    results = {"medium/hf-fp32/cpu-4t": result(12.0), "medium/ct2-int8/cpu-4t": result(3.0, wer=0.08), "ner/spacy/cpu": result(0.001, wer=None),
               "medium/broken/cpu-4t": {"error": "boom"}}
    baseline = {"medium/ct2-int8/cpu-4t": {"median_s": "4.0", "wer": "0.05"}}
    text = b.render_report("r9", "tag", INFO, results, baseline)
    line = next(ln for ln in text.splitlines() if ln.startswith("| medium/ct2-int8/cpu-4t"))
    assert "4.0x" in line  # 12 s / 3 s against the fp32 reference
    assert "-25% latency" in line and "+3.0 pts WER" in line  # 3.0 vs 4.0 s, 8% vs 5% WER
    assert "| 0.001 |" in next(ln for ln in text.splitlines() if ln.startswith("| ner/spacy/cpu")) and "ERROR: boom" in text
    assert "new" in next(ln for ln in text.splitlines() if ln.startswith("| medium/hf-fp32/cpu-4t"))  # not in the baseline


def test_save_run_writes_json_latest_and_baseline_report(tmp_path):
    report = b.render_report("r1", "baseline", INFO, {"medium/hf-fp32/cpu-4t": result(12.0)})
    path = b.save_run(tmp_path, "r1", "baseline", "t1", INFO, {"n": 10}, {"medium/hf-fp32/cpu-4t": result(12.0)}, report)
    assert path.exists() and (tmp_path / "latest.md").read_text() == report and (tmp_path / "BASELINE.md").exists()
    b.save_run(tmp_path, "r2", "experiment", "t2", INFO, {}, {"medium/hf-fp32/cpu-4t": result(6.0)}, "other")
    assert (tmp_path / "BASELINE.md").read_text() == report  # only a baseline-tagged run updates BASELINE.md


def test_report_compares_a_dynamic_window_config_with_its_full_window_twin():
    results = {"medium/ct2-int8-dynwin/cpu-4t": result(0.8, wer=0.06), "medium/brand-new/cpu-4t": result(1.0)}
    baseline = {"medium/ct2-int8/cpu-4t": {"median_s": "2.5", "wer": "0.05"}}
    text = b.render_report("r9", "tag", INFO, results, baseline)
    line = next(ln for ln in text.splitlines() if ln.startswith("| medium/ct2-int8-dynwin/cpu-4t"))
    assert "-68% latency" in line and "+1.0 pts WER (vs medium/ct2-int8/cpu-4t)" in line
    assert "new" in next(ln for ln in text.splitlines() if ln.startswith("| medium/brand-new"))


def test_paired_wer_table_separates_a_real_difference_from_an_identical_config():
    refs = ["contact rhein one three two"] * 60
    hyps = {"ref": list(refs), "same": list(refs),
            "worse": [r if i % 3 else "contact rhein one three" for i, r in enumerate(refs)]}  # one word dropped in a third of the clips
    rows = {r["config"]: r for r in b.paired_wer_table(refs, hyps, "ref")}
    assert rows["ref"]["wer"] == 0 and "diff" not in rows["ref"]
    assert rows["same"]["verdict"] == "no clear difference" and rows["same"]["identical_to_ref"] == 1.0
    assert rows["worse"]["verdict"] == "worse" and rows["worse"]["diff_lo"] > 0
    assert rows["worse"]["wer"] == pytest.approx(20 / 300) and rows["worse"]["identical_to_ref"] == pytest.approx(2 / 3)
    assert "worse" in b.render_paired(list(rows.values()), "ref", "title")
