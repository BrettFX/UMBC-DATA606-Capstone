#!/usr/bin/env python3
"""Local web dashboard for the long-running jobs: NER annotation and ASR fine-tuning (standard library only).

    python scripts/ner/monitor.py [--port 8765]      then open http://localhost:8765

NER annotation: reads the newest progress_<signature>.json written by scripts/run_ner_annotate.py.
ASR fine-tuning: parses the newest res/logs/asr_train_*.log (trainer progress bar, loss and validation
lines) and reads test_scores.json once scoring is done. Both show the tail of their log plus `nvidia-smi`.
It only reads files, so stopping it never affects a job. Binds to localhost.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANN_DIR = REPO_ROOT / "data" / "processed" / "ner_dataset" / "annotations"
LOG_DIR = REPO_ROOT / "res" / "logs"
MODELS_DIR = REPO_ROOT / "models"


def newest(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime) if directory.exists() else []
    return files[-1] if files else None


def gpu() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split(",")
        return dict(util=int(out[0]), mem_used=int(out[1]), mem_total=int(out[2]), temp=int(out[3]))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def process_running(needle: str) -> bool:
    """True if some process's command line contains `needle` (and is not this dashboard)."""
    for entry in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            cmd = entry.read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if needle in cmd and "monitor.py" not in cmd and "pgrep" not in cmd:
            return True
    return False


def ner_status() -> dict | None:
    prog_path, log_path = newest(ANN_DIR, "progress_*.json"), newest(ANN_DIR, "run_*.log")
    if not prog_path:
        return None
    prog = json.loads(prog_path.read_text())
    try:
        os.kill(prog["pid"], 0)
        alive = True
    except OSError:
        alive = False
    return {"progress": prog, "process_alive": alive, "heartbeat_age_s": time.time() - prog_path.stat().st_mtime,
            "log_tail": log_path.read_text().splitlines()[-14:] if log_path else []}


_BAR = re.compile(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+), *([\d.]+)(s/it|it/s)\]")
_LOSS = re.compile(r"\{'loss': '([\d.]+)'.*?'epoch': '([\d.]+)'\}")
_EVAL = re.compile(r"\{'eval_loss': '([\d.]+)', 'eval_wer': '([\d.]+)'.*?'epoch': '([\d.]+)'\}")
_TEST = re.compile(r"test WER ([\d.]+) CER ([\d.]+)")


def _seconds(clock: str) -> float | None:
    if "?" in clock:
        return None
    parts = [int(x) for x in clock.split(":")]
    return float(sum(p * 60 ** i for i, p in enumerate(reversed(parts))))


def asr_status() -> dict | None:
    """Training status parsed from the newest ASR run log (the trainer logs its own progress bar)."""
    log_path = newest(LOG_DIR, "asr_train_*.log")
    if not log_path:
        return None
    with open(log_path, "rb") as f:  # the tail holds everything needed; progress-bar redraws make the file long
        f.seek(max(0, log_path.stat().st_size - 3_000_000))
        text = f.read().decode(errors="ignore")
    lines = [ln for ln in re.split(r"[\r\n]+", text) if ln.strip()]
    bars = _BAR.findall(text)
    # The training bar redraws every step, so its total is the most common one; other bars (weight loading,
    # validation passes) have a different total and far fewer redraws.
    totals = Counter(int(b[1]) for b in bars)
    total = max(totals, key=lambda t: (totals[t], t)) if totals else None
    train_bars = [b for b in bars if int(b[1]) == total]
    step = elapsed = eta = rate = None
    if train_bars:
        step, _, elapsed, eta, speed, unit = train_bars[-1]
        step, elapsed, eta = int(step), _seconds(elapsed), _seconds(eta)
        rate = float(speed) if unit == "s/it" else 1 / float(speed)
    losses = [(float(e), float(l)) for l, e in _LOSS.findall(text)]
    evals = [{"epoch": float(e), "wer": float(w), "loss": float(l)} for l, w, e in _EVAL.findall(text)]
    done = "finished OK" in text
    gave_up = "gave up after" in text
    alive = process_running("run_asr_train.py")
    status = "finished" if done else "error" if gave_up or (not alive and step is not None and step < (total or 0)) else "running"
    test = _TEST.search(text)
    scores = None
    for path in sorted(MODELS_DIR.glob("*-atc-finetuned-full/test_scores.json"), key=lambda p: p.stat().st_mtime)[-1:]:
        scores = json.loads(path.read_text())
    return {"status": status, "step": step, "total": total, "epoch": losses[-1][0] if losses else None,
            "elapsed_s": elapsed, "eta_s": eta, "s_per_step": rate, "loss_curve": [round(l, 4) for _, l in losses[-200:]],
            "latest_loss": losses[-1][1] if losses else None, "evals": evals,
            "attempts": len(re.findall(r"=== attempt \d+", text)), "process_alive": alive,
            "heartbeat_age_s": time.time() - log_path.stat().st_mtime, "log_file": log_path.name,
            "test": {"wer": float(test.group(1)), "cer": float(test.group(2))} if test else None, "test_scores": scores,
            "log_tail": [ln[:220] for ln in lines if not _BAR.search(ln)][-12:]}


def status() -> dict:
    return {"ner": ner_status(), "asr": asr_status(), "gpu": gpu(), "now": time.time()}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Run Monitor</title><style>
:root{--bg:#fafafa;--fg:#1d2329;--mut:#66727d;--card:#fff;--line:#d9dee3;--acc:#2a78d6;--ok:#2e7d32;--bad:#c62828;--warn:#b26a00}
@media(prefers-color-scheme:dark){:root{--bg:#14181c;--fg:#e6e9ec;--mut:#97a3ad;--card:#1d2329;--line:#333c44;--acc:#6aa9f0;--ok:#6cc070;--bad:#ef6a6a;--warn:#e0a040}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg)}main{max-width:960px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px}h2{font-size:16px;margin:22px 0 2px}.sub{color:var(--mut);font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.tile b{display:block;font-size:22px}.tile span{color:var(--mut);font-size:12px}
.bar{height:14px;background:var(--line);border-radius:7px;overflow:hidden}.bar div{height:100%;background:var(--acc);width:0}
.pill{display:inline-block;padding:1px 10px;border-radius:10px;font-size:12px;font-weight:600;color:#fff;margin-left:6px}pre{margin:0;font-size:12px;white-space:pre-wrap;word-break:break-all;color:var(--mut)}
.row{display:flex;gap:8px;align-items:center;margin:3px 0;font-size:13px}.row i{display:block;height:10px;background:var(--acc);border-radius:3px}.row span:first-child{width:90px}svg{width:100%;height:60px}
table{border-collapse:collapse;font-size:13px}td,th{padding:2px 14px 2px 0;text-align:left}th{color:var(--mut);font-weight:500}
</style></head><body><main><h1>Run Monitor</h1><div class="sub" id="gpu"></div>
<section id="asr" hidden><h2>ASR fine-tuning <span id="a_pill" class="pill"></span></h2><div class="sub" id="a_sub"></div>
<div class="card"><div class="bar"><div id="a_bar"></div></div><div class="sub" id="a_lbl" style="margin-top:6px"></div></div>
<div class="card tiles" id="a_tiles"></div>
<div class="card"><b>Training loss</b><svg id="a_spark" viewBox="0 0 300 60" preserveAspectRatio="none"></svg><div class="sub" id="a_sparklbl"></div></div>
<div class="card"><b>Validation WER by epoch</b><div id="a_evals" class="sub"></div></div>
<div class="card" id="a_testcard" hidden><b>Test-set scores (full test split)</b><div id="a_test"></div></div>
<div class="card"><b>Log</b><pre id="a_log"></pre></div></section>
<section id="ner" hidden><h2>NER annotation <span id="pill" class="pill"></span></h2><div class="sub" id="sub"></div>
<div class="card"><div class="bar"><div id="bar"></div></div><div class="sub" id="barlbl" style="margin-top:6px"></div></div>
<div class="card tiles" id="tiles"></div>
<div class="card"><b>Seconds per 256-utterance chunk</b><svg id="spark" viewBox="0 0 300 60" preserveAspectRatio="none"></svg><div class="sub" id="sparklbl"></div></div>
<div class="card"><b>Entities labeled so far</b><div id="labels"></div></div>
<div class="card"><b>Log</b><pre id="log"></pre></div></section>
<div class="sub" id="none" hidden>No annotation or training run found yet.</div></main>
<script>
const $=id=>document.getElementById(id), fmt=s=>{if(s==null)return"–";const h=Math.floor(s/3600),m=Math.round(s%3600/60);return h?h+"h "+m+"m":m+"m"};
const tiles=(el,T)=>{$(el).innerHTML=T.map(t=>`<div class="tile"><b>${t[1]}</b><span>${t[0]}</span></div>`).join("")};
const pill=(el,txt,col)=>{$(el).textContent=txt;$(el).style.background=col};
function spark(svg,lbl,vals,unit){if(vals.length>1){const mx=Math.max(...vals)*1.1,mn=Math.min(...vals)*0.9,pts=vals.map((v,i)=>(i/(vals.length-1)*300)+","+(58-(v-mn)/(mx-mn||1)*56)).join(" ");
$(svg).innerHTML=`<polyline fill="none" stroke="var(--acc)" stroke-width="1.5" points="${pts}"/>`;$(lbl).textContent=vals.length+" points · latest "+vals[vals.length-1]+unit+" · range "+Math.min(...vals)+"–"+Math.max(...vals)+unit}
else $(lbl).textContent="waiting for data…"}
function renderNer(n,g){const p=n.progress;$("ner").hidden=false;
const stale=n.heartbeat_age_s>1500,dead=n.process_alive===false&&p.status==="running";
const st=p.status==="finished"?["finished","var(--ok)"]:p.status==="error"||dead?["stopped / error","var(--bad)"]:stale?["stalled?","var(--warn)"]:["running","var(--ok)"];
pill("pill",st[0],st[1]);$("sub").textContent=p.model+" · signature "+p.signature+" · last update "+Math.round(n.heartbeat_age_s)+"s ago (a chunk takes ~13 min)"+(dead?" · PROCESS NOT FOUND":"");
const pct=100*p.annotated/p.pool_size;$("bar").style.width=pct+"%";$("barlbl").textContent=p.annotated.toLocaleString()+" / "+p.pool_size.toLocaleString()+" utterances ("+pct.toFixed(1)+"%)";
tiles("tiles",[["Remaining",p.status==="finished"?"done":fmt(p.eta_seconds)],["Rate",p.rate_utt_per_s?(1/p.rate_utt_per_s).toFixed(2)+" s/utt":"–"],["Done this run",p.this_run_done.toLocaleString()],
["Valid",(100*p.ok_rate).toFixed(1)+"%"],["Needed retry",(100*p.retry_rate).toFixed(1)+"%"],["GPU",g?g.util+"% · "+(g.mem_used/1024).toFixed(1)+"/"+(g.mem_total/1024).toFixed(0)+" GB":"n/a"]]);
spark("spark","sparklbl",p.chunk_seconds||[],"s");
const L=Object.entries(p.labels||{}).sort((a,b)=>b[1]-a[1]),m=Math.max(1,...L.map(x=>x[1]));
$("labels").innerHTML=L.map(([k,v])=>`<div class="row"><span>${k}</span><i style="width:${v/m*60}%"></i><span>${v.toLocaleString()}</span></div>`).join("");
$("log").textContent=n.log_tail.join("\\n")}
function renderAsr(a,g){$("asr").hidden=false;
const stale=a.status==="running"&&a.heartbeat_age_s>900;
const st=a.status==="finished"?["finished","var(--ok)"]:a.status==="error"?["stopped / error","var(--bad)"]:stale?["stalled?","var(--warn)"]:["running","var(--ok)"];
pill("a_pill",st[0],st[1]);$("a_sub").textContent=a.log_file+" · attempt "+a.attempts+" · last log update "+Math.round(a.heartbeat_age_s)+"s ago";
const done=a.status==="finished"&&a.step!=null,pct=a.step&&a.total?100*a.step/a.total:(done?100:0);$("a_bar").style.width=pct+"%";
$("a_lbl").textContent=a.step!=null?"step "+a.step+" / "+a.total+" · epoch "+(a.epoch??"–")+" ("+pct.toFixed(1)+"%)":"starting…";
const bw=a.evals.length?Math.min(...a.evals.map(e=>e.wer)):null;
tiles("a_tiles",[["Remaining",done?"done":fmt(a.eta_s)],["Speed",a.s_per_step?a.s_per_step.toFixed(2)+" s/step":"–"],["Elapsed",fmt(a.elapsed_s)],
["Latest loss",a.latest_loss!=null?a.latest_loss.toFixed(3):"–"],["Best val WER",bw!=null?(100*bw).toFixed(1)+"%":"–"],["GPU",g?g.util+"% · "+(g.mem_used/1024).toFixed(1)+"/"+(g.mem_total/1024).toFixed(0)+" GB":"n/a"]]);
spark("a_spark","a_sparklbl",a.loss_curve,"");
$("a_evals").innerHTML=a.evals.length?"<table><tr><th>epoch</th><th>val WER</th><th>val loss</th></tr>"+a.evals.map(e=>`<tr><td>${e.epoch}</td><td>${(100*e.wer).toFixed(2)}%</td><td>${e.loss.toFixed(3)}</td></tr>`).join("")+"</table>":"after the first epoch…";
$("a_testcard").hidden=!a.test;if(a.test){const s=a.test_scores;$("a_test").innerHTML=`<div class="tile"><b>WER ${(100*a.test.wer).toFixed(1)}% · CER ${(100*a.test.cer).toFixed(1)}%</b><span>${s?s.n+" test utterances":""}</span></div>`+(s?"<table><tr><th>source</th><th>n</th><th>WER</th><th>CER</th></tr>"+Object.entries(s.by_source).map(([k,v])=>`<tr><td>${k}</td><td>${v.n}</td><td>${(100*v.wer).toFixed(1)}%</td><td>${(100*v.cer).toFixed(1)}%</td></tr>`).join("")+"</table>":"")}
$("a_log").textContent=a.log_tail.join("\\n")}
async function tick(){let d;try{d=await (await fetch("/status")).json()}catch(e){$("gpu").textContent="monitor offline";return}
const g=d.gpu;$("gpu").textContent=g?"GPU "+g.util+"% · "+(g.mem_used/1024).toFixed(1)+" / "+(g.mem_total/1024).toFixed(0)+" GB · "+g.temp+"°C":"GPU: n/a";
$("none").hidden=!!(d.ner||d.asr);if(d.asr)renderAsr(d.asr,g);if(d.ner)renderNer(d.ner,g)}
tick();setInterval(tick,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/status"):
            body, ctype = json.dumps(status()).encode(), "application/json"
        else:
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the terminal quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"Monitoring {ANN_DIR} and {LOG_DIR}\nOpen http://localhost:{args.port}  (Ctrl-C to stop; the jobs are unaffected)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
