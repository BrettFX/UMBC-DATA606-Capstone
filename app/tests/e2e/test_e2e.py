"""Browser end-to-end tests (Playwright). Starts the real Streamlit server on a free port.

Run: python -m pytest app/tests/e2e -q      (screenshots land in app/tests/e2e/shots/)
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[3]
SHOTS = Path(__file__).parent / "shots"
SHOTS.mkdir(exist_ok=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/main.py", "--server.headless", "true",
                             "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://localhost:{port}"
    for _ in range(60):
        try:
            if urllib.request.urlopen(url + "/_stcore/health", timeout=1).read() == b"ok":
                break
        except Exception:
            time.sleep(1)
    else:
        proc.kill()
        pytest.fail("server did not start")
    yield url
    proc.terminate()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        yield b
        b.close()


@pytest.fixture
def page(browser, server):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.errors = errors
    pg.goto(server)
    yield pg
    ctx.close()


def enter(page):
    page.get_by_role("button", name="Enter ClearanceIQ").click()
    expect(page.locator("[data-testid=stAlertContentWarning]").first).to_be_visible(timeout=20000)


def test_splash_animates_and_fades_into_home(page):
    btn = page.get_by_role("button", name="Enter ClearanceIQ")
    expect(btn).to_be_visible(timeout=20000)
    expect(page.locator("#ciq-canvas")).to_be_visible()
    page.wait_for_timeout(2500)
    page.screenshot(path=SHOTS / "splash_a.png")
    page.wait_for_timeout(1500)
    page.screenshot(path=SHOTS / "splash_b.png")
    # canvas actually animates: two frames differ
    a, b = (SHOTS / "splash_a.png").read_bytes(), (SHOTS / "splash_b.png").read_bytes()
    assert a != b
    assert page.locator("[data-testid=stSidebar]").count() == 0 or not page.locator("[data-testid=stSidebar]").is_visible()
    btn.click()
    page.wait_for_timeout(700)  # mid-fade: splash still in the DOM and partially transparent
    op = page.evaluate("getComputedStyle(document.getElementById('ciq-splash')||document.body).opacity")
    assert 0 < float(op) < 1, f"expected mid-fade opacity, got {op}"
    page.screenshot(path=SHOTS / "splash_fading.png")
    expect(page.locator("[data-testid=stAlertContentWarning]").first).to_be_visible(timeout=15000)
    assert page.locator("#ciq-splash").count() == 0
    page.wait_for_timeout(2200)
    page.screenshot(path=SHOTS / "home.png")
    assert not page.errors


def settle(page):
    """Wait for the auto-run analysis of the default sample so later widget clicks are not lost to a rerun."""
    expect(page.locator(".ciq-live .ciq-w").first).to_be_visible(timeout=90000)
    page.wait_for_timeout(500)


def test_sample_list_contains_validation_and_longest(page):
    enter(page)
    settle(page)
    page.get_by_label("Labeled sample").click()
    opts = page.get_by_role("option").all_inner_texts()
    assert len(opts) >= 15
    assert any("validation" in o for o in opts)
    assert any("LONGEST" in o for o in opts)


def test_default_sample_end_to_end(page):
    enter(page)
    settle(page)
    # tabs: the audio analysis tab comes first and is the one shown
    names = page.get_by_role("tab").all_inner_texts()
    assert names[0] == "Audio analysis", names
    assert page.get_by_role("tab", selected=True).first.inner_text() == "Audio analysis"
    expect(page.locator(".js-plotly-plot").first).to_be_visible(timeout=20000)
    assert page.locator(".js-plotly-plot").count() >= 2
    page.get_by_role("tab", name="Transcript & entities").click()
    expect(page.get_by_text("Clearance", exact=True).first).to_be_visible(timeout=30000)
    expect(page.locator("iframe").first).to_be_attached(timeout=30000)
    expect(page.get_by_text("Highlighted transcript")).to_be_visible()
    page.get_by_role("tab", name="Evidence").click()
    expect(page.get_by_text("WER", exact=True).first).to_be_visible()
    page.get_by_role("tab", name="Explain (AI)").click()
    expect(page.get_by_text("This feature is off until")).to_be_visible()
    assert page.get_by_role("button", name="Explain this transmission").is_disabled()
    page.screenshot(path=SHOTS / "explain_disabled.png")
    assert not page.errors


def pick_option(page, text):
    settle(page)
    page.get_by_label("Labeled sample").click()
    page.get_by_role("option").filter(has_text=re.compile(text)).first.click()


def test_longest_sample_is_split_and_transcribed(page):
    enter(page)
    pick_option(page, "LONGEST")
    expect(page.get_by_text(re.compile("Split into \\d+ segments"))).to_be_visible(timeout=60000)
    expect(page.locator(".ciq-live .ciq-seg").first).to_be_visible(timeout=120000)
    assert page.locator(".ciq-live .ciq-seg").count() >= 2
    assert page.locator(".ciq-live .ciq-w[data-s]").count() > 50          # every word of both segments is timed
    page.screenshot(path=SHOTS / "longest.png", full_page=True)
    assert not page.errors


def test_validation_sample_runs(page):
    enter(page)
    pick_option(page, "validation")
    page.get_by_role("tab", name="Evidence").click()
    expect(page.get_by_text(re.compile("comes from the validation split"))).to_be_visible(timeout=90000)
    page.screenshot(path=SHOTS / "validation.png", full_page=True)


def test_bad_upload_shows_friendly_error(page, tmp_path):
    enter(page)
    page.locator("[data-testid=stSidebar]").get_by_text("Upload file").click()
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"this is not audio" * 50)
    page.locator("input[type=file]").set_input_files(str(bad))
    expect(page.get_by_text("could not be decoded")).to_be_visible(timeout=30000)
    assert page.locator("text=Traceback").count() == 0


def test_good_upload_works(page):
    enter(page)
    page.locator("[data-testid=stSidebar]").get_by_text("Upload file").click()
    wav = next((ROOT / "app/samples").glob("atcosim-train-000666.wav"))
    page.locator("input[type=file]").set_input_files(str(wav))
    page.get_by_role("button", name="Analyze").click()
    expect(page.locator(".ciq-live .ciq-w").first).to_be_visible(timeout=90000)


def test_model_cards(page):
    enter(page)
    settle(page)
    page.get_by_role("tab", name="Models").click()
    # the card for the model selected in the sidebar is shown first
    expect(page.get_by_text("Aviation entity extraction (NER) for ATC transcripts")).to_be_visible(timeout=20000)
    expect(page.get_by_text("selected in sidebar")).to_be_visible()
    page.screenshot(path=SHOTS / "model_card_overview.png", full_page=True)
    page.get_by_role("tab", name="Metrics").click()
    expect(page.get_by_text("Held-out (adjudicated)").first).to_be_visible()
    page.screenshot(path=SHOTS / "model_card_metrics.png", full_page=True)
    page.get_by_role("tab", name="Training").click()
    expect(page.get_by_text("Class balance")).to_be_visible()
    page.get_by_text("Compare entity models").click()
    expect(page.get_by_text("Which should I pick?")).to_be_visible(timeout=20000)
    page.screenshot(path=SHOTS / "model_card_compare.png", full_page=True)
    page.get_by_text("Speech recognition", exact=True).click()
    expect(page.locator(".mc-title")).to_have_text("Whisper medium (ATC fine-tuned, LoRA)", timeout=20000)
    page.screenshot(path=SHOTS / "model_card_asr.png", full_page=True)
    assert not page.errors


def test_playhead_follows_audio_and_click_seeks(page):
    enter(page)
    settle(page)
    page.get_by_role("tab", name="Audio analysis").click()
    expect(page.locator(".js-plotly-plot").first).to_be_visible(timeout=20000)
    page.wait_for_function("document.querySelectorAll('.ciq-playhead').length >= 2", timeout=10000)

    def audio_ready():
        return page.evaluate("(() => { const a = document.querySelector('audio'); return a && isFinite(a.duration) && a.duration > 0 })()")
    for _ in range(40):
        if audio_ready():
            break
        page.wait_for_timeout(250)
    dur = page.evaluate("document.querySelector('audio').duration")

    def head_left(i):
        return page.evaluate(f"""(() => {{ const l = document.querySelectorAll('.ciq-playhead')[{i}];
            return l.style.display === 'none' ? null : parseFloat(l.style.left) }})()""")
    page.evaluate(f"document.querySelector('audio').currentTime = {dur * 0.3}")
    page.wait_for_timeout(300)
    a = head_left(0)
    page.evaluate(f"document.querySelector('audio').currentTime = {dur * 0.8}")
    page.wait_for_timeout(300)
    b = head_left(0)
    assert a is not None and b is not None and b > a, (a, b)
    assert head_left(1) is not None                       # spectrogram has a playhead too
    page.screenshot(path=SHOTS / "playhead.png", full_page=True)

    # click-to-seek: click the middle of the waveform plot area
    box = page.locator(".js-plotly-plot").first.bounding_box()
    page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    page.wait_for_timeout(300)
    t = page.evaluate("document.querySelector('audio').currentTime")
    assert abs(t - dur * 0.5) < dur * 0.2, (t, dur)

    # real playback moves the line
    page.evaluate("document.querySelector('audio').currentTime = 0")
    page.evaluate("document.querySelector('audio').muted = true")
    played = page.evaluate("document.querySelector('audio').play().then(() => true).catch(() => false)")
    if played:
        page.wait_for_timeout(1200)
        assert page.evaluate("document.querySelector('audio').currentTime") > 0.5
        assert (head_left(0) or 0) > 0
    assert not page.errors


def test_live_transcript_visible_on_every_tab_and_highlights_words(page):
    enter(page)
    settle(page)
    words = page.locator(".ciq-live .ciq-w[data-s]")
    n = words.count()
    assert n >= 5
    # the transcript sits directly under the audio player and stays visible whichever tab is open
    pos = page.evaluate("""(() => { const a = document.querySelector('[data-testid=stAudio]').getBoundingClientRect();
        const t = document.querySelector('.ciq-live').getBoundingClientRect(); return [a.bottom, t.top] })()""")
    assert 0 <= pos[1] - pos[0] < 120, pos
    for tab in ("Transcript & entities", "Evidence", "Models", "Audio analysis"):
        page.get_by_role("tab", name=tab).click()
        expect(page.locator(".ciq-live").first).to_be_visible()
    # seeking into the middle of the third word lights up exactly that word
    info = page.evaluate("""(() => { const w = document.querySelectorAll('.ciq-live .ciq-w[data-s]')[2];
        return [parseFloat(w.dataset.s), parseFloat(w.dataset.e), w.textContent] })()""")
    mid = (info[0] + info[1]) / 2
    for _ in range(40):
        if page.evaluate("(() => { const a = document.querySelector('audio'); return a && isFinite(a.duration) && a.duration > 0 })()"):
            break
        page.wait_for_timeout(250)
    page.evaluate(f"document.querySelector('audio').currentTime = {mid}")
    page.wait_for_timeout(400)
    on = page.locator(".ciq-live .ciq-w.on")
    expect(on).to_have_count(1)
    assert on.inner_text() == info[2]
    assert page.locator(".ciq-live .ciq-w.done").count() == 2           # the two words before it are marked as spoken
    page.screenshot(path=SHOTS / "live_transcript.png", full_page=True)
    # clicking a later word jumps the player to it
    target = page.evaluate("""(() => { const w = document.querySelectorAll('.ciq-live .ciq-w[data-s]')[n - 1]; return parseFloat(w.dataset.s) })()""".replace("n - 1", str(n - 1)))
    words.nth(n - 1).click()
    page.wait_for_timeout(300)
    t = page.evaluate("document.querySelector('audio').currentTime")
    assert abs(t - target) < 0.3, (t, target)
    assert not page.errors


def test_header_is_sticky(page):
    enter(page)
    settle(page)
    header = page.locator(".st-key-ciq_header")
    expect(header).to_be_visible()
    before = header.bounding_box()["y"]
    assert before > 20, before                      # not pinned yet: it sits below the top padding
    page.get_by_role("tab", name="Audio analysis").click()
    page.evaluate("document.querySelector('[data-testid=stMain]').scrollTop = 900")
    page.wait_for_timeout(400)
    after = header.bounding_box()
    assert 0 <= after["y"] < 5, after                # pinned at the very top of the viewport (not scrolled away)
    assert page.evaluate("document.querySelector('[data-testid=stMain]').scrollTop") > 300
    expect(page.locator(".ciq-brand-name")).to_be_in_viewport()
    page.screenshot(path=SHOTS / "sticky_header.png")
    assert not page.errors


def test_word_highlight_works_when_an_older_script_is_already_in_the_page(browser, server):
    """Regression: a page that already ran an earlier version of the sync script must still get word highlighting and click-to-seek."""
    ctx = browser.new_context(viewport={"width": 1300, "height": 900})
    pg = ctx.new_page()
    pg.add_init_script("window.__ciqPlayhead = true; window.__ciqToken = 'older-version';")
    pg.goto(server)
    enter(pg)
    settle(pg)
    pg.evaluate("document.querySelector('audio').muted = true")
    info = pg.evaluate("""(() => { const w = document.querySelectorAll('.ciq-live .ciq-w[data-s]')[3];
        return [parseFloat(w.dataset.s), parseFloat(w.dataset.e), w.textContent] })()""")
    for _ in range(40):
        if pg.evaluate("(() => { const a = document.querySelector('audio'); return a && isFinite(a.duration) && a.duration > 0 })()"):
            break
        pg.wait_for_timeout(250)
    pg.evaluate(f"document.querySelector('audio').currentTime = {(info[0] + info[1]) / 2}")
    pg.wait_for_timeout(400)
    expect(pg.locator(".ciq-live .ciq-w.on")).to_have_count(1)
    pg.locator(".ciq-live .ciq-w[data-s]").nth(8).click()
    pg.wait_for_timeout(300)
    target = pg.evaluate("parseFloat(document.querySelectorAll('.ciq-live .ciq-w[data-s]')[8].dataset.s)")
    assert abs(pg.evaluate("document.querySelector('audio').currentTime") - target) < 0.3
    ctx.close()


def test_word_highlight_during_real_playback(browser, server):
    """Plays the clip for real (muted, autoplay allowed) and checks that the highlight moves through several words."""
    ctx = browser.new_context(viewport={"width": 1300, "height": 900})
    pg = ctx.new_page()
    pg.goto(server)
    enter(pg)
    settle(pg)
    pg.evaluate("document.querySelector('audio').muted = true")
    assert pg.evaluate("document.querySelector('audio').play().then(() => true).catch(() => false)") is True
    seen = set()
    for _ in range(40):
        pg.wait_for_timeout(120)
        seen |= set(pg.locator(".ciq-live .ciq-w.on").all_inner_texts())
    assert len(seen) >= 6, seen
    ctx.close()
