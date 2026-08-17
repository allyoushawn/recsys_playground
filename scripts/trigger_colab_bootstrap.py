#!/usr/bin/env python3
"""
trigger_colab_bootstrap.py

Automatically opens the Colab bootstrap notebook, connects to a GPU runtime,
runs all cells, and returns the cloudflared SSH hostname.

Architecture:
  - Chrome is launched via subprocess (NOT by Playwright) so Google sees a real
    browser and doesn't block sign-in or challenge auth.
  - Playwright connects to it via CDP (remote debugging port) for automation.
  - Session is stored in ~/.colab_chrome_profile/ between runs.

First-time setup (one-time only):
    python3 scripts/trigger_colab_bootstrap.py --setup
    → Opens real Chrome to accounts.google.com.
    → Sign in, then close Chrome. Session saved for all future runs.

Normal usage:
    python3 scripts/trigger_colab_bootstrap.py
    → Prints hostname to stdout on success.
    → Exits with code 1 on failure.
    → Writes viewport PNGs to /tmp/colab_trigger_<timestamp>_NN_<step>.png for review (disable: --no-screenshots).
    → After runtime connects, checks the Colab UI for the requested accelerator token (e.g. T4) and fails fast if missing.

Concurrent / isolated sessions:
    By default all sessions share CDP port 9222 and profile ~/.colab_chrome_profile,
    and launching a new session force-kills whatever Chrome is already on that port.
    To run a second, isolated session (e.g. while a first session's Chrome is being
    kept alive via --keep-chrome) without touching the first session's Chrome:
        python3 scripts/trigger_colab_bootstrap.py --cdp-port 9223 --profile-dir ~/.colab_chrome_profile_2
    A profile dir used for the first time has no saved Google session -- run once
    with --setup --profile-dir <same dir> before normal use with that dir.

Requirements:
    pip3 install playwright
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# Set True via --debug-dialog (logs md-text-button snapshots when clicking OK in modals).
_DEBUG_DIALOG = True

# Progress screenshots for humans/agents (viewport PNGs). Disable with --no-screenshots.
_PROCESS_SCREENSHOTS = True
_SCREENSHOT_DIR = Path("/tmp")
_RUN_SCREENSHOT_STAMP = ""
_SCREENSHOT_SEQ = 0
# Minimum seconds between runtime-wait / ntfy-wait viewport screenshots (CLI: --screenshot-interval).
_SCREENSHOT_INTERVAL_SEC = 1.0
# Set when Change-runtime-type → Save succeeds (used if DOM never exposes GPU label text).
_RUNTIME_TYPE_SAVE_GPU: str | None = None

# ── Configuration ─────────────────────────────────────────────────────────────

NOTEBOOK_URL = (
    "https://colab.research.google.com/github/allyoushawn/recsys_playground"
    "/blob/main/notebooks/ad_hoc/colab_ssh_bootstrap.ipynb?flush_cache=true"
)

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
COLAB_PROFILE = Path.home() / ".colab_chrome_profile"
CDP_PORT = 9222
CDP_URL = f"http://localhost:{CDP_PORT}"

RUNTIME_CONNECT_TIMEOUT = 180
HOSTNAME_WAIT_TIMEOUT = 300
HOSTNAME_RE = re.compile(r'([\w-]+\.trycloudflare\.com)')

NTFY_POLL_INTERVAL = 10  # seconds between ntfy.sh polls

# Load ntfy topic from local secrets file (never hardcode in public repo)
_SECRETS_FILE = Path(__file__).parent.parent / ".colab_secrets"

def _load_ntfy_topic() -> str:
    if not _SECRETS_FILE.exists():
        raise RuntimeError(
            f"Secrets file not found: {_SECRETS_FILE}\n"
            "Create it with: NTFY_TOPIC=colab-ssh-<your-uuid>\n"
            "Also add NTFY_TOPIC as a Colab Secret (left panel → key icon) so the notebook can read it."
        )
    for line in _SECRETS_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("NTFY_TOPIC="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"NTFY_TOPIC not found in {_SECRETS_FILE}")

NTFY_TOPIC = _load_ntfy_topic()
SSH_KEY = Path.home() / ".ssh" / "colab_key"


def _screenshot_run_init() -> None:
    global _RUN_SCREENSHOT_STAMP, _SCREENSHOT_SEQ
    _RUN_SCREENSHOT_STAMP = time.strftime("%Y%m%d_%H%M%S")
    _SCREENSHOT_SEQ = 0


async def _screenshot_process(page, step: str) -> None:
    """Save a viewport PNG under _SCREENSHOT_DIR for post-run review (e.g. read_image on /tmp)."""
    if not _PROCESS_SCREENSHOTS:
        return
    if page is None:
        return
    global _SCREENSHOT_SEQ
    _SCREENSHOT_SEQ += 1
    safe = re.sub(r"[^\w\-.]+", "_", (step or "step").strip())[:55].strip("_") or "step"
    fname = f"colab_trigger_{_RUN_SCREENSHOT_STAMP}_{_SCREENSHOT_SEQ:03d}_{safe}.png"
    out = _SCREENSHOT_DIR / fname
    try:
        _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(out), full_page=False)
        print(f"[trigger] Screenshot ({step}) → {out}", file=sys.stderr)
    except Exception as exc:
        print(f"[trigger] Screenshot failed ({step}): {exc}", file=sys.stderr)


def _gpu_match_keyword(gpu_type: str) -> str | None:
    """Return a short token to match in Colab UI (e.g. 'T4 GPU' -> 'T4'). None = skip check."""
    s = (gpu_type or "").strip()
    if not s or re.match(r"(?i)^none$", s):
        return None
    # Strip trailing " GPU" / "gpu"
    s = re.sub(r"(?i)\s*gpu\s*$", "", s).strip()
    return s or None


async def _runtime_gpu_probe_text(page) -> str:
    """Collect visible strings Colab uses for accelerator / RAM (best-effort)."""
    try:
        return await page.evaluate(
            """() => {
              function textDeep(root, maxLen) {
                let out = '';
                function walk(n) {
                  if (!n || out.length >= maxLen) return;
                  if (n.nodeType === 3) out += n.textContent || '';
                  if (n.shadowRoot) walk(n.shadowRoot);
                  const kids = n.children || [];
                  for (let i = 0; i < kids.length && out.length < maxLen; i++) walk(kids[i]);
                }
                walk(root);
                return out.slice(0, maxLen);
              }
              const chunks = [];
              const u = document.querySelector('colab-usage-display');
              if (u) {
                chunks.push(u.textContent || '');
                chunks.push(textDeep(u, 6000));
              }
              const host = document.querySelector('colab-connect-button');
              if (host && host.shadowRoot) {
                const b = host.shadowRoot.querySelector('button');
                if (b) chunks.push(b.textContent || '');
                chunks.push(textDeep(host, 8000));
              }
              const rts = document.querySelectorAll('colab-runtime-status, colab-env-details, colab-footer, colab-toolbar, #top-toolbar, #toolbar-area');
              rts.forEach(el => chunks.push(textDeep(el, 8000)));
              document.querySelectorAll('[aria-label]').forEach(el => {
                const a = el.getAttribute('aria-label') || '';
                if (/(T4|L4|A100|H100|V100|TPU|GPU|Python)/i.test(a)) chunks.push(a);
              });
              const bar = document.querySelector('colab-header, colab-title-bar, [class*=\"colab\"]');
              if (bar) chunks.push((bar.textContent || '').slice(0, 400));
              const body = (document.body && document.body.innerText) ? document.body.innerText : '';
              chunks.push(body.slice(0, 50000));
              return chunks.join(' | ').slice(0, 62000);
            }"""
        )
    except Exception:
        return ""


async def _assert_runtime_matches_gpu(page, gpu_type: str) -> None:
    """Raise if the visible UI does not appear to show the requested accelerator.

    Colab often shows RAM / runtime chips before the accelerator label (e.g. T4)
    finishes painting — poll for up to 75s before failing.
    """
    key = _gpu_match_keyword(gpu_type)
    if not key:
        print("[trigger] GPU verify skipped (--gpu None or empty).", file=sys.stderr)
        return
    pat = re.compile(r"\b" + re.escape(key) + r"\b", re.I)
    deadline = asyncio.get_event_loop().time() + 75
    attempt = 0
    while asyncio.get_event_loop().time() < deadline:
        attempt += 1
        blob = await _runtime_gpu_probe_text(page)
        if blob.strip() and pat.search(blob):
            print(f"[trigger] GPU verify OK: found {key!r} in UI probe (attempt {attempt}).", file=sys.stderr)
            return
        # Accessibility / pierced tree (sometimes innerText lags behind shadow labels).
        try:
            loc = page.get_by_text(re.compile(rf"\b{re.escape(key)}\b", re.I)).first
            if await loc.is_visible(timeout=800):
                print(f"[trigger] GPU verify OK: visible text match for {key!r} (attempt {attempt}).", file=sys.stderr)
                return
        except Exception:
            pass
        await asyncio.sleep(1)

    blob = await _runtime_gpu_probe_text(page)
    global _RUNTIME_TYPE_SAVE_GPU
    if _RUNTIME_TYPE_SAVE_GPU:
        saved_key = _gpu_match_keyword(_RUNTIME_TYPE_SAVE_GPU)
        if saved_key and saved_key.lower() == key.lower():
            print(
                f"[trigger] GPU verify SOFT OK: DOM never showed {key!r}, but runtime-type dialog Save "
                f"succeeded for {_RUNTIME_TYPE_SAVE_GPU!r}.",
                file=sys.stderr,
            )
            return
    if not blob.strip():
        raise RuntimeError(
            "Runtime GPU verify failed: no readable UI text after 75s (probe still empty)."
        )
    await _screenshot_process(page, f"gpu_verify_FAIL_expected_{key}")
    raise RuntimeError(
        f"Runtime GPU verify failed: never saw accelerator {key!r} within 75s. Last probe excerpt: {blob[:500]!r}..."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _launch_chrome(url: str) -> subprocess.Popen:
    """Launch real Chrome (no Playwright shim) with our dedicated profile."""
    # Kill any Chrome already holding our debug port
    subprocess.run(
        f"lsof -ti tcp:{CDP_PORT} | xargs kill -9",
        shell=True, capture_output=True
    )
    time.sleep(1)

    # Remove stale SingletonLock left by a previous Chrome run
    for lock in ["SingletonLock", "SingletonSocket", "SingletonCookie"]:
        lock_path = COLAB_PROFILE / lock
        if lock_path.exists():
            lock_path.unlink()
            print(f"[trigger] Removed stale {lock}.", file=sys.stderr)

    # Mark the previous session as clean so Chrome skips the "Restore page?" infobar.
    # SIGKILL leaves exit_type="Crashed"; patching it to "Normal" suppresses the prompt.
    local_state = COLAB_PROFILE / "Local State"
    if local_state.exists():
        try:
            state = json.loads(local_state.read_text())
            for profile_name in state.get("profile", {}).get("info_cache", {}):
                state["profile"]["info_cache"][profile_name]["exit_type"] = "Normal"
            local_state.write_text(json.dumps(state))
        except Exception as e:
            print(f"[trigger] Could not patch Local State: {e}", file=sys.stderr)

    cmd = [
        CHROME_BIN,
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={COLAB_PROFILE}",
        "--no-first-run",
        "--no-default-browser-check",
        url,
    ]
    print(f"[trigger] Launching Chrome → {url}", file=sys.stderr)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def _wait_for_cdp(timeout=45):
    """Wait until Chrome's CDP endpoint is accepting connections."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{CDP_URL}/json/version", timeout=2)
            print(f"[trigger] CDP ready.", file=sys.stderr)
            return
        except Exception:
            await asyncio.sleep(1)
    raise RuntimeError(f"Chrome CDP not available at {CDP_URL} after {timeout}s")


def _colab_page(ctx):
    for p in ctx.pages:
        if "colab.research.google.com" in p.url:
            return p
    return ctx.pages[0] if ctx.pages else None


async def _handle_modal(page, detect_text, click_label, timeout_ms=3_000):
    try:
        if not await page.locator(f"text={detect_text}").first.is_visible(timeout=timeout_ms):
            return False
    except Exception:
        return False
    print(f"[trigger] Modal '{detect_text}' → clicking '{click_label}'", file=sys.stderr)
    try:
        await page.get_by_role("button", name=click_label).first.click(timeout=5_000)
        await asyncio.sleep(1)
        return True
    except Exception:
        try:
            await page.locator(f"text={click_label}").first.click(timeout=3_000)
            return True
        except Exception:
            print(f"[trigger] Warning: could not click '{click_label}'", file=sys.stderr)
            return False


# Colab "trusted notebook" warning is localized. Match body fragments + button labels.
_TRUSTED_NOTEBOOK_HINTS = [
    ("not authored by Google", "Run anyway"),
    ("並非由 Google", "仍要執行"),  # zh-TW (and many zh variants in title)
    ("并非由 Google", "仍然运行"),  # zh-CN — title / button wording may vary by region
]


async def _dismiss_trusted_notebook_warning(page):
    """Dismiss 'notebook not authored by Google' (GitHub / third-party) dialog in any locale."""
    for detect, click in _TRUSTED_NOTEBOOK_HINTS:
        if await _handle_modal(page, detect, click, timeout_ms=2_000):
            return True
    # Last resort: primary action label only (avoids missing new English variants).
    try:
        btn = page.get_by_role(
            "button",
            name=re.compile(
                r"Run anyway|仍要執行|仍然运行|继续运行|繼續執行|"
                r"Ejecutar de todos modos|Ejecutar de todas formas|Ausführen|Exécuter quand même",
                re.I,
            ),
        ).first
        if await btn.is_visible(timeout=2_000):
            print("[trigger] Trusted-notebook dialog → Run anyway (regex match).", file=sys.stderr)
            await btn.click(timeout=5_000)
            await asyncio.sleep(1)
            return True
    except Exception:
        pass
    return False


async def _dismiss_disallowed_warning(page) -> bool:
    """Dismiss Colab's 'you may be executing disallowed code' warning → 'Continue anyway'."""
    return await _handle_modal(page, "executing code that is disallowed", "Continue anyway", timeout_ms=2_000)


async def _handle_all_modals(ctx, page):
    await _dismiss_trusted_notebook_warning(page)
    await _dismiss_disallowed_warning(page)
    for text in ["Dismiss", "Got it", "No thanks"]:
        try:
            btn = page.locator(f"text={text}").first
            if await btn.is_visible(timeout=500):
                await btn.click()
        except Exception:
            pass
    return _colab_page(ctx) or page


async def _click_connect(page):
    # Click the colab-connect-button element directly (center of element lands on
    # the "Connect" text area, not the ▼ chevron). This is the approach that
    # worked reliably for T4 GPU connections.
    for selector, label in [
        ("colab-connect-button", "colab-connect-button"),
        ("#connect", "#connect"),
    ]:
        try:
            btn = page.locator(selector).first
            await btn.click(timeout=8_000)
            print(f"[trigger] Clicked {label}.", file=sys.stderr)
            return
        except Exception:
            pass
    try:
        btn = page.get_by_role("button", name=re.compile(r"Connect", re.I)).first
        await btn.click(timeout=5_000)
        print("[trigger] Clicked Connect (ARIA role).", file=sys.stderr)
    except Exception:
        print("[trigger] Warning: Connect not clicked — may already be connected.", file=sys.stderr)


async def _open_connect_dropdown(page) -> bool:
    """Open the Connect-button dropdown (the ▼ chevron). Returns True on success."""
    opened = await page.evaluate("""() => {
        const host = document.querySelector('colab-connect-button');
        if (!host || !host.shadowRoot) return false;
        const btns = Array.from(host.shadowRoot.querySelectorAll('button, [role="button"]'));
        const trigger = btns.find(b => b.getAttribute('aria-haspopup')) || btns[btns.length - 1];
        if (!trigger) return false;
        trigger.click();
        return true;
    }""")
    if not opened:
        cb = page.locator("colab-connect-button").first
        box = await cb.bounding_box()
        if box:
            await page.mouse.click(box["x"] + box["width"] - 8, box["y"] + box["height"] / 2)
            return True
        return False
    return True


def _disconnect_runtime_dialog_locators(page):
    """Locators for session-termination / runtime-change confirmation surfaces.

    Colab uses <md-dialog> whose content is in shadow DOM — role-based locators
    with has_text filter can't pierce the shadow, so we include md-dialog directly.
    """
    hint = re.compile(
        r"Are you sure you want to continue|"
        r"terminate your current session|"
        r"Changing runtime attributes may terminate",
        re.I,
    )
    return (
        page.locator("md-dialog").last,           # topmost md-dialog (most specific)
        page.locator("md-dialog"),                 # any md-dialog
        page.get_by_role("alertdialog").filter(has_text=hint),
        page.get_by_role("dialog").filter(has_text=hint),
    )


async def _runtime_disconnect_dialog_visible(page) -> bool:
    """True when the disconnect / dangerous-action confirmation is still on screen."""
    for loc in _disconnect_runtime_dialog_locators(page):
        try:
            if await loc.count() == 0:
                continue
            if await loc.first.is_visible(timeout=500):
                return True
        except Exception:
            continue
    return False


async def _dialog_roots_for_disconnect(page):
    """Playwright locators for each visible confirmation dialog root (nth)."""
    roots = []
    for loc in _disconnect_runtime_dialog_locators(page):
        try:
            n = await loc.count()
            for i in range(n):
                el = loc.nth(i)
                if await el.is_visible(timeout=400):
                    roots.append(el)
        except Exception:
            continue
    return roots


async def _debug_dump_runtime_dialog(page) -> None:
    """Log visible md-* button labels (innerText) for UI-automation debugging."""
    try:
        data = await page.evaluate("""() => {
          const out = [];
          function walk(root) {
            for (const el of root.querySelectorAll('*')) {
              if (el.shadowRoot) walk(el.shadowRoot);
              const tag = el.tagName ? el.tagName.toLowerCase() : '';
              if (tag === 'md-text-button' || tag === 'md-filled-button' || tag === 'md-outlined-button') {
                const r = el.getBoundingClientRect();
                const t = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, 80);
                if (r.width > 0 && r.height > 0) {
                  out.push({tag, t, x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)});
                }
              }
            }
          }
          walk(document);
          return out;
        }""")
        blob = json.dumps(data, ensure_ascii=False)
        print(f"[trigger] DEBUG md-buttons: {blob[:12000]}", file=sys.stderr)
    except Exception as exc:
        print(f"[trigger] DEBUG dump failed: {exc}", file=sys.stderr)


async def _click_ok_in_dialog(page) -> bool:
    """Click OK on Colab disconnect / dangerous-action confirmation.

    Scopes Playwright clicks to dialog|alertdialog that contain session-termination
    copy — avoids clicking unrelated OK buttons (e.g. secrets / share prompts).
    Verifies the confirmation surface closed when possible.
    """
    if _DEBUG_DIALOG:
        await _debug_dump_runtime_dialog(page)

    roots = await _dialog_roots_for_disconnect(page)

    async def _try_click(locator, note: str) -> bool:
        try:
            if await locator.count() == 0:
                return False
            btn = locator.first
            if not await btn.is_visible(timeout=2500):
                return False
            await btn.click(timeout=8000)
            print(f"[trigger] OK clicked ({note}).", file=sys.stderr)
            await asyncio.sleep(0.45)
            return not await _runtime_disconnect_dialog_visible(page)
        except Exception:
            return False

    # 1) Scoped Playwright: md-text-button / md-filled-button / ARIA role
    for root in roots:
        chain = [
            (root.locator("md-text-button").filter(has_text=re.compile(r"^\s*OK\s*$", re.I)), "scoped md-text-button ^OK$"),
            (root.locator("md-filled-button").filter(has_text=re.compile(r"^\s*OK\s*$", re.I)), "scoped md-filled-button ^OK$"),
            (root.get_by_role("button", name=re.compile(r"^\s*ok\s*$", re.I)), "scoped ARIA button OK"),
            (root.locator("md-text-button").filter(has_text="OK"), "scoped md-text-button substring OK"),
        ]
        for loc, note in chain:
            if await _try_click(loc, note):
                return True

        # 2) Right-to-left scan: OK is usually the rightmost affirmative control
        try:
            mds = root.locator("md-text-button")
            n = await mds.count()
            for idx in range(n - 1, -1, -1):
                btn = mds.nth(idx)
                if not await btn.is_visible(timeout=600):
                    continue
                txt = (await btn.inner_text()).strip()
                if re.match(r"(?i)^ok\s*$", txt):
                    await btn.click(timeout=8000)
                    print("[trigger] OK clicked (scoped md-text-button by inner_text).", file=sys.stderr)
                    await asyncio.sleep(0.45)
                    if not await _runtime_disconnect_dialog_visible(page):
                        return True
        except Exception:
            pass

    # 3) JS: collect md-text-button / md-filled-button centers; prefer /^ok$/i, else rightmost \bok\b
    js_result = await page.evaluate("""() => {
      const cands = [];
      function walk(root) {
        for (const el of root.querySelectorAll('*')) {
          if (el.shadowRoot) walk(el.shadowRoot);
          const tag = el.tagName ? el.tagName.toLowerCase() : '';
          if (tag === 'md-text-button' || tag === 'md-filled-button' || tag === 'md-outlined-button') {
            const t = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
            const r = el.getBoundingClientRect();
            if (r.width > 0 && r.height > 0) {
              cands.push({t, x: r.x + r.width / 2, y: r.y + r.height / 2, left: r.x});
            }
          }
        }
      }
      walk(document);
      const exact = cands.filter(c => /^ok$/i.test(c.t));
      const pool = exact.length ? exact : cands.filter(c => /\\bok\\b/i.test(c.t));
      pool.sort((a, b) => b.left - a.left);
      const pick = pool[0];
      return pick
        ? {found: true, x: pick.x, y: pick.y, labels: cands.map(c => c.t).slice(0, 16)}
        : {found: false, labels: cands.map(c => c.t).slice(0, 16)};
    }""")
    if js_result.get("found"):
        print(
            f"[trigger] OK via JS material scan → ({js_result['x']:.0f}, {js_result['y']:.0f}); labels={js_result.get('labels')}",
            file=sys.stderr,
        )
        await page.mouse.click(js_result["x"], js_result["y"])
        await asyncio.sleep(0.45)
        if not await _runtime_disconnect_dialog_visible(page):
            return True
        print("[trigger] JS OK click did not dismiss disconnect dialog — fallbacks.", file=sys.stderr)

    if _DEBUG_DIALOG:
        print(f"[trigger] DEBUG: after JS, dialog_visible={await _runtime_disconnect_dialog_visible(page)}", file=sys.stderr)

    # 4) Legacy: plain <button role=button> scan (textContent)
    result = await page.evaluate("""() => {
        const found = [];
        function collect(root) {
            for (const el of root.querySelectorAll('*')) {
                if (el.shadowRoot) collect(el.shadowRoot);
                const tag = el.tagName ? el.tagName.toLowerCase() : '';
                const role = el.getAttribute ? el.getAttribute('role') : '';
                if (tag === 'button' || role === 'button') {
                    const text = el.textContent.trim();
                    const r = el.getBoundingClientRect();
                    if (r.width > 0 && r.height > 0) {
                        found.push({text, x: r.x + r.width / 2, y: r.y + r.height / 2});
                    }
                }
            }
        }
        collect(document);
        const ok = found.find(b => /^ok$/i.test(b.text));
        return ok ? {found: true, ...ok} : {found: false, all: found.map(b => b.text)};
    }""")
    if result.get("found"):
        print(f"[trigger] OK via legacy <button> scan at ({result['x']:.0f}, {result['y']:.0f}).", file=sys.stderr)
        await page.mouse.click(result["x"], result["y"])
        await asyncio.sleep(0.45)
        if not await _runtime_disconnect_dialog_visible(page):
            return True

    await page.keyboard.press("Enter")
    print("[trigger] Pressed Enter as last-resort OK.", file=sys.stderr)
    await asyncio.sleep(0.45)
    dismissed = not await _runtime_disconnect_dialog_visible(page)
    if not dismissed and _DEBUG_DIALOG:
        await _debug_dump_runtime_dialog(page)
    return dismissed


async def _disconnect_runtime(page) -> bool:
    """Disconnect and delete the current runtime via the Connect dropdown.

    Clicks the 'Disconnect and delete runtime' menu item, then handles the
    resulting confirmation dialog by clicking OK.
    Returns True if disconnect was triggered, False if no runtime was live.
    """
    if not await _runtime_is_live(page):
        return False

    opened = await _open_connect_dropdown(page)
    if not opened:
        return False
    await asyncio.sleep(0.8)

    disconnect = page.get_by_role("menuitem", name="Disconnect and delete runtime")
    try:
        if await disconnect.is_visible(timeout=3_000):
            # Skip if disabled (e.g. runtime still connecting — option not available yet)
            disabled = await disconnect.get_attribute("aria-disabled")
            if disabled == "true":
                print("[trigger] 'Disconnect and delete runtime' is disabled — skipping.", file=sys.stderr)
                await page.keyboard.press("Escape")
                return False
            await disconnect.click()
            print("[trigger] Clicked 'Disconnect and delete runtime' menu item.", file=sys.stderr)
            # The menu item triggers a confirmation dialog — click OK to confirm.
            await asyncio.sleep(1.5)
            for attempt in range(1, 4):
                clicked_ok = await _click_ok_in_dialog(page)
                if clicked_ok or not await _runtime_disconnect_dialog_visible(page):
                    break
                print(
                    f"[trigger] Disconnect dialog still visible after OK attempt {attempt} — retrying.",
                    file=sys.stderr,
                )
                await asyncio.sleep(0.7)
            if await _runtime_disconnect_dialog_visible(page):
                await page.keyboard.press("Enter")
                print("[trigger] Pressed Enter as fallback for dialog OK.", file=sys.stderr)
                await asyncio.sleep(0.4)
            await asyncio.sleep(2)
            return True
    except Exception as exc:
        print(f"[trigger] _disconnect_runtime error: {exc}", file=sys.stderr)
    await page.keyboard.press("Escape")
    return False


async def _set_runtime_type_gpu(page, gpu_type: str = "T4 GPU", high_ram: bool = False) -> bool:
    """Disconnect any running runtime, then set the hardware accelerator type.

    Strategy:
    1. If a runtime is live, disconnect it first (avoids the confirmation overlay
       that appears when changing type with an active session).
    2. Open Connect dropdown → Change runtime type → select GPU radio → optionally
       select High-RAM shape → Save.

    Returns True if the runtime type was set, False otherwise.
    """
    try:
        # ── Step 0: disconnect existing runtime to avoid confirmation dialog ──
        if await _runtime_is_live(page):
            print("[trigger] Disconnecting existing runtime before type change...", file=sys.stderr)
            await _disconnect_runtime(page)
            # Wait for runtime to fully disconnect
            for _ in range(10):
                if not await _runtime_is_live(page):
                    break
                await asyncio.sleep(1)
            print("[trigger] Runtime disconnected.", file=sys.stderr)

        # ── Step 1: open the Connect-button dropdown ──────────────────────────
        # The ▼ chevron is inside colab-connect-button's shadow DOM.
        # Strategy: inspect the shadow root, log what we find, then click the
        # rightmost button (the dropdown trigger).
        debug_info = await page.evaluate("""() => {
            const host = document.querySelector('colab-connect-button');
            if (!host) return {found: false, reason: 'no colab-connect-button'};
            const root = host.shadowRoot;
            if (!root) return {found: false, reason: 'no shadowRoot (closed mode?)'};
            const btns = Array.from(root.querySelectorAll('button, [role="button"]'));
            return {
                found: true,
                count: btns.length,
                labels: btns.map(b => b.getAttribute('aria-label') || b.textContent.trim().slice(0,30)),
                ariaHaspopup: btns.map(b => b.getAttribute('aria-haspopup')),
            };
        }""")
        print(f"[trigger] Connect button shadow DOM: {debug_info}", file=sys.stderr)

        opened = False

        # Try clicking the button with aria-haspopup (the dropdown trigger)
        opened = await page.evaluate("""() => {
            const host = document.querySelector('colab-connect-button');
            if (!host || !host.shadowRoot) return false;
            const btns = Array.from(host.shadowRoot.querySelectorAll('button, [role="button"]'));
            // Prefer aria-haspopup button; fall back to last button
            const trigger = btns.find(b => b.getAttribute('aria-haspopup')) || btns[btns.length - 1];
            if (!trigger) return false;
            trigger.click();
            return true;
        }""")

        if not opened:
            # Fallback: click the right edge of colab-connect-button (where ▼ lives)
            print("[trigger] JS open failed; clicking right edge of connect button.", file=sys.stderr)
            cb = page.locator("colab-connect-button").first
            box = await cb.bounding_box()
            if box:
                await page.mouse.click(box["x"] + box["width"] - 8, box["y"] + box["height"] / 2)
                opened = True

        if not opened:
            print("[trigger] Could not open Connect dropdown — skipping GPU type set.", file=sys.stderr)
            return False
        print("[trigger] Opened Connect dropdown.", file=sys.stderr)
        await asyncio.sleep(0.8)

        # ── Step 2: click "Change runtime type" in the dropdown ──────────────
        # get_by_role uses the accessibility tree which pierces shadow DOM.
        change_rt = page.get_by_role("menuitem", name="Change runtime type")
        try:
            await change_rt.click(timeout=5_000)
            print("[trigger] Clicked 'Change runtime type'.", file=sys.stderr)
        except Exception:
            await page.keyboard.press("Escape")
            print("[trigger] 'Change runtime type' menuitem not found.", file=sys.stderr)
            return False
        await asyncio.sleep(1.5)
        await _screenshot_process(page, "gpu_dialog_opened")

        # ── Step 3: select the GPU radio button ───────────────────────────────
        # Use md-dialog-scoped md-radio elements directly.
        # Avoid page.get_by_role("radio") page-wide — it can match links outside
        # the dialog that cause a page refresh (dismissing the dialog).
        selected = False

        # Primary: use Playwright locators which pierce open shadow DOMs.
        # JS document.querySelector cannot find elements inside shadow roots.
        # Log what radio elements are visible to diagnose on failure.
        all_radios = page.get_by_role("radio")
        radio_count = await all_radios.count()
        radio_names = []
        for i in range(radio_count):
            try:
                r = all_radios.nth(i)
                if await r.is_visible(timeout=300):
                    nm = await r.evaluate("el => el.getAttribute('aria-label') || el.textContent.trim() || el.value || '?'")
                    radio_names.append(nm)
            except Exception:
                pass
        print(f"[trigger] Visible radios: {radio_names}", file=sys.stderr)

        for selector, label in [
            ("mat-radio-button", "mat-radio-button"),
            ("mat-radio-group mat-radio-button", "mat-radio-group > mat-radio-button"),
            ("mat-dialog-container mat-radio-button", "mat-dialog-container mat-radio-button"),
        ]:
            try:
                loc = page.locator(selector).filter(has_text=gpu_type)
                if await loc.count() > 0 and await loc.first.is_visible(timeout=1_000):
                    await loc.first.click()
                    print(f"[trigger] Selected radio via '{selector}'.", file=sys.stderr)
                    selected = True
                    break
            except Exception:
                pass

        if not selected:
            # Fallback: page-wide ARIA radio with exact name match (safe, won't match links)
            try:
                radio = page.get_by_role("radio", name=re.compile(r"^\s*" + re.escape(gpu_type) + r"\s*$", re.I))
                if await radio.first.is_visible(timeout=2_000):
                    await radio.first.click()
                    print(f"[trigger] Selected radio via ARIA name.", file=sys.stderr)
                    selected = True
            except Exception:
                pass

        if not selected:
            await page.keyboard.press("Escape")
            print(f"[trigger] '{gpu_type}' radio not found — check account tier.", file=sys.stderr)
            return False
        await asyncio.sleep(0.5)
        await _screenshot_process(page, "gpu_radio_selected")

        # ── Step 3c: select High-RAM shape if requested ───────────────────────
        if high_ram:
            hr_selected = False
            for selector, label in [
                ("mat-radio-button", "mat-radio-button"),
                ("mat-radio-group mat-radio-button", "mat-radio-group > mat-radio-button"),
                ("mat-dialog-container mat-radio-button", "mat-dialog-container mat-radio-button"),
            ]:
                try:
                    loc = page.locator(selector).filter(has_text=re.compile(r"High.RAM", re.I))
                    if await loc.count() > 0 and await loc.first.is_visible(timeout=1_000):
                        await loc.first.click()
                        print("[trigger] Selected High-RAM radio.", file=sys.stderr)
                        hr_selected = True
                        break
                except Exception:
                    pass

            if not hr_selected:
                # Fallback: ARIA radio name match
                try:
                    radio = page.get_by_role("radio", name=re.compile(r"High.RAM", re.I))
                    if await radio.first.is_visible(timeout=2_000):
                        await radio.first.click()
                        print("[trigger] Selected High-RAM radio via ARIA.", file=sys.stderr)
                        hr_selected = True
                except Exception:
                    pass

            if not hr_selected:
                print("[trigger] High-RAM radio not found (not available on this account/tier) — continuing with standard RAM.", file=sys.stderr)
            await asyncio.sleep(0.5)

        # ── Step 3b: handle unexpected "Disconnect and delete runtime" confirmation ─
        # With the disconnect-first strategy (Step 0), this dialog should not appear.
        # Keep as a safety net for edge cases where disconnect didn't fully complete.
        confirmed = False
        deadline = asyncio.get_event_loop().time() + 4
        while asyncio.get_event_loop().time() < deadline:
            dialog_visible = await page.evaluate("""() => {
                function hasText(root, text) {
                    for (const el of root.querySelectorAll('*')) {
                        if (el.shadowRoot && hasText(el.shadowRoot, text)) return true;
                        if (el.children.length === 0 && el.textContent.includes(text)) return true;
                    }
                    return false;
                }
                return hasText(document, 'Are you sure you want to continue');
            }""")
            if dialog_visible:
                print("[trigger] Unexpected disconnect confirmation — clicking OK.", file=sys.stderr)
                clicked_ok = await _click_ok_in_dialog(page)
                if not clicked_ok:
                    await page.keyboard.press("Enter")
                await asyncio.sleep(1.5)
                confirmed = True
                break
            await asyncio.sleep(0.5)
        if not confirmed:
            print("[trigger] No disconnect confirmation dialog — continuing.", file=sys.stderr)

        # ── Step 4: click Save ────────────────────────────────────────────────
        # Use page-wide role search — the dialog is not <md-dialog> so scoping fails.
        save_btn = page.get_by_role("button", name="Save")
        try:
            await save_btn.click(timeout=5_000)
            print(f"[trigger] Runtime type saved ({gpu_type}).", file=sys.stderr)
        except Exception:
            await page.keyboard.press("Escape")
            print("[trigger] Save button not found.", file=sys.stderr)
            return False
        global _RUNTIME_TYPE_SAVE_GPU
        _RUNTIME_TYPE_SAVE_GPU = gpu_type
        await asyncio.sleep(1.5)
        await _screenshot_process(page, "gpu_save_clicked")
        return True

    except Exception as exc:
        print(f"[trigger] _set_runtime_type_gpu error: {exc}", file=sys.stderr)
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass
        return False


async def _connect_button_label(page) -> str:
    """Lowercase label from colab-connect-button shadow (empty if unavailable)."""
    try:
        return await page.evaluate("""() => {
            const host = document.querySelector('colab-connect-button');
            if (!host || !host.shadowRoot) return '';
            const btn = host.shadowRoot.querySelector('button');
            return btn ? btn.textContent.trim().toLowerCase() : '';
        }""")
    except Exception:
        return ""


async def _runtime_is_live(page) -> bool:
    """Return True when a runtime session is connected or actively provisioning.

    Colab can show L4 in the runtime chip while the top-right is still idle
    "Connect" — `colab-runtime-status` alone must not count as live.
    """
    btn_text = await _connect_button_label(page)
    if btn_text == "connect":
        return False

    for sel in ("colab-usage-display", "text=RAM", "[data-connected='true']"):
        try:
            el = await page.query_selector(sel)
            if el and await el.is_visible():
                return True
        except Exception:
            pass

    for text in ("Resuming session", "Connecting", "Initializing"):
        try:
            el = await page.query_selector(f"text={text}")
            if el and await el.is_visible():
                return True
        except Exception:
            pass

    if btn_text:
        return True
    return False



async def _ensure_connect_if_idle(ctx, page, note: str) -> object:
    """If UI is still on idle Connect, click it (GPU type can show L4 without a session)."""
    page = _colab_page(ctx) or page
    label = await _connect_button_label(page)
    if label != "connect":
        return page
    print(f"[trigger] {note} — clicking Connect (runtime type was set but no session).", file=sys.stderr)
    await _click_connect(page)
    await asyncio.sleep(4)
    page = _colab_page(ctx) or page
    return await _handle_all_modals(ctx, page)




async def _runtime_past_provisioning(page) -> bool:
    """Return False while Colab UI still shows VM allocation (cells cannot run yet).

    `colab-runtime-status` may be visible while the footer still says e.g. "Allocating runtime"
    and the header shows "… Connecting" — we must not treat that as "connected".
    """
    try:
        return await page.evaluate(
            r"""() => {
              const b = document.body ? document.body.innerText : '';
              if (/Allocating runtime/i.test(b)) return false;
              if (/Resuming session/i.test(b)) return false;
              const head = b.slice(0, 3500);
              if (/\u2026\s*Connecting|\.\.\.\s*Connecting/i.test(head)) return false;
              return true;
            }"""
        )
    except Exception:
        return True


async def _wait_runtime_connected(ctx):
    print(f"[trigger] Waiting for runtime (up to {RUNTIME_CONNECT_TIMEOUT}s)...", file=sys.stderr)
    connected_selectors = [
        "colab-usage-display", "text=RAM",
        "[data-connected='true']", "colab-runtime-status",
    ]
    deadline = asyncio.get_event_loop().time() + RUNTIME_CONNECT_TIMEOUT
    last_log = 0.0
    last_shot = 0.0
    while asyncio.get_event_loop().time() < deadline:
        page = _colab_page(ctx)
        if page:
            for sel in connected_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el and await el.is_visible():
                        if not await _runtime_past_provisioning(page):
                            continue
                        print(f"[trigger] Runtime live (via '{sel}').", file=sys.stderr)
                        return page
                except Exception:
                    pass
            now = asyncio.get_event_loop().time()
            # Retry connect if "Unable to connect" toast is visible.
            try:
                if await page.locator("text=Unable to connect to the runtime").first.is_visible(timeout=500):
                    print("[trigger] 'Unable to connect' toast detected — retrying connect click.", file=sys.stderr)
                    await _click_connect(page)
                    await asyncio.sleep(3)
            except Exception:
                pass
            # Viewport screenshot every _SCREENSHOT_INTERVAL_SEC while waiting.
            if now - last_shot >= _SCREENSHOT_INTERVAL_SEC:
                try:
                    elapsed = int(now - (deadline - RUNTIME_CONNECT_TIMEOUT))
                    await _screenshot_process(page, f"still_waiting_runtime_{elapsed}s")
                except Exception:
                    pass
                last_shot = now
            # Textual progress log every 10s (avoid noisy stderr).
            if now - last_log >= 10.0:
                try:
                    btn_text = await page.evaluate("""() => {
                        const host = document.querySelector('colab-connect-button');
                        if (!host || !host.shadowRoot) return 'no-shadow';
                        const btn = host.shadowRoot.querySelector('button');
                        return btn ? btn.textContent.trim() : 'no-btn';
                    }""")
                    print(f"[trigger] Still waiting — connect button: '{btn_text}', page URL: {page.url[:80]}", file=sys.stderr)
                except Exception:
                    pass
                last_log = now
        await asyncio.sleep(1)
    # Screenshots at timeout (numbered + legacy path for older tooling)
    page = _colab_page(ctx)
    if page:
        try:
            await _screenshot_process(page, "runtime_connect_timeout")
            await page.screenshot(path="/tmp/colab_timeout.png", full_page=False)
            print("[trigger] Screenshot also saved to /tmp/colab_timeout.png", file=sys.stderr)
        except Exception:
            pass
    raise RuntimeError(f"Runtime did not connect within {RUNTIME_CONNECT_TIMEOUT}s.")


async def _handle_drive_auth_background(ctx) -> None:
    """
    Background task: handle all Drive authorization dialogs.

    Two forms:
    1. Inline Colab dialog — "Permit this notebook to access your Google Drive
       files?" with a "Connect to Google Drive" button (appears on the Colab page).
    2. Google OAuth popup tab — accounts.google.com/o/oauth2 with an Allow button
       (opened after clicking the inline dialog, if credentials need re-consent).
    """
    ALLOW_RE = re.compile(
        r"Allow|Continue|確認|授予|Autoriser|Zulassen|Permitir|Consenti",
        re.I,
    )
    seen_urls = set()
    while True:
        # ── 1. Inline Colab "Connect to Google Drive" dialog ──────────────────
        colab_page = _colab_page(ctx)
        if colab_page:
            for label in ["Connect to Google Drive", "连接到 Google Drive", "連接至 Google 雲端硬碟"]:
                try:
                    btn = colab_page.get_by_role("button", name=re.compile(label, re.I)).first
                    if await btn.is_visible(timeout=500):
                        print(f"[trigger] Drive inline dialog: clicking '{label}'.", file=sys.stderr)
                        await btn.click()
                        await asyncio.sleep(2)
                        break
                except Exception:
                    pass

        # ── 3. Colab Secrets dialogs ──────────────────────────────────────────
        # Two forms:
        # a) "Share notebook" warning (appears when userdata.get() is called on a
        #    GitHub-URL notebook) — has an OK button and "secret" or "Share" text.
        # b) Per-secret "Grant access" prompt — has a "Grant access" button.
        if colab_page:
            # Form a: "Share notebook" modal with OK
            try:
                share_title = colab_page.locator("text=Share notebook").first
                if await share_title.is_visible(timeout=500):
                    ok_btn = colab_page.get_by_role("button", name=re.compile(r"^ok$", re.I)).first
                    if await ok_btn.is_visible(timeout=2_000):
                        print("[trigger] Secrets 'Share notebook' warning: clicking OK.", file=sys.stderr)
                        await ok_btn.click()
                        await asyncio.sleep(1)
            except Exception:
                pass
            # Form b: per-secret Grant access prompt
            for label in ["Grant access", "Allow"]:
                try:
                    btn = colab_page.get_by_role("button", name=re.compile(label, re.I)).first
                    if await btn.is_visible(timeout=500):
                        secrets_hint = colab_page.locator("text=secret").first
                        if await secrets_hint.is_visible(timeout=500):
                            print(f"[trigger] Colab secrets grant dialog: clicking '{label}'.", file=sys.stderr)
                            await btn.click()
                            await asyncio.sleep(1)
                            break
                except Exception:
                    pass

        # ── 2. OAuth popup tab (accounts.google.com/o/oauth2) ─────────────────
        for pg in list(ctx.pages):
            url = pg.url
            if "accounts.google.com" in url and "oauth" in url.lower() and url not in seen_urls:
                seen_urls.add(url)
                print(f"[trigger] Drive OAuth tab: {url[:80]}", file=sys.stderr)
                await asyncio.sleep(2)  # let page settle
                try:
                    btn = pg.get_by_role("button", name=ALLOW_RE).first
                    if await btn.is_visible(timeout=8_000):
                        await btn.click()
                        print("[trigger] Drive OAuth: clicked Allow.", file=sys.stderr)
                        continue
                except Exception:
                    pass
                for label in ["Allow", "Continue", "Sign in"]:
                    try:
                        b = pg.locator(f"text={label}").first
                        if await b.is_visible(timeout=2_000):
                            await b.click()
                            print(f"[trigger] Drive OAuth: clicked '{label}'.", file=sys.stderr)
                            break
                    except Exception:
                        pass

        await asyncio.sleep(2)


async def _hostname_from_frame(frame):
    """Scan one frame's DOM + open shadow roots for trycloudflare hostname."""
    try:
        raw = await frame.evaluate(
            r"""
            () => {
                const re = /[\w-]+\.trycloudflare\.com/;
                function walk(node) {
                    if (!node) return null;
                    if (node.nodeType === 3) {
                        const m = (node.textContent || '').match(re);
                        if (m) return m[0];
                    }
                    if (node.shadowRoot) {
                        const r = walk(node.shadowRoot);
                        if (r) return r;
                    }
                    const kids = node.childNodes;
                    if (kids) {
                        for (let i = 0; i < kids.length; i++) {
                            const r = walk(kids[i]);
                            if (r) return r;
                        }
                    }
                    return null;
                }
                const body = document.body;
                if (!body) return null;
                return walk(body);
            }
            """
        )
        if not raw:
            return None
        m = HOSTNAME_RE.search(raw)
        return m.group(1).strip() if m else None
    except Exception:
        return None


async def _wait_for_hostname_ntfy(ctx, start_epoch: int) -> str:
    """Poll ntfy.sh for the hostname POSTed by the bootstrap notebook.

    While polling, captures Colab viewport screenshots every `_SCREENSHOT_INTERVAL_SEC`
    so long ntfy waits are still reviewable.
    """
    import json
    import urllib.request

    def _poll_ntfy_once() -> str | None:
        try:
            url = f"https://ntfy.sh/{NTFY_TOPIC}/json?since={start_epoch}&poll=1"
            with urllib.request.urlopen(url, timeout=10) as resp:
                lines = resp.read().decode().strip().splitlines()
            for line in reversed(lines):
                try:
                    msg = json.loads(line)
                    hostname = msg.get("message", "").strip()
                    if HOSTNAME_RE.match(hostname):
                        return hostname
                except Exception:
                    pass
        except Exception as e:
            print(f"[trigger] ntfy.sh poll error: {e}", file=sys.stderr)
        return None

    print(f"[trigger] Polling ntfy.sh for hostname (up to {HOSTNAME_WAIT_TIMEOUT}s)...", file=sys.stderr)
    deadline = time.time() + HOSTNAME_WAIT_TIMEOUT
    next_poll = time.time()
    last_shot = 0.0
    t0 = time.time()
    while time.time() < deadline:
        now = time.time()
        if now >= next_poll:
            host = await asyncio.to_thread(_poll_ntfy_once)
            if host:
                print(f"[trigger] Hostname received: {host}", file=sys.stderr)
                return host
            next_poll = now + NTFY_POLL_INTERVAL
        page = _colab_page(ctx)
        if page:
            await _dismiss_trusted_notebook_warning(page)
            await _dismiss_disallowed_warning(page)
            if _PROCESS_SCREENSHOTS and (now - last_shot >= _SCREENSHOT_INTERVAL_SEC):
                await _screenshot_process(page, f"ntfy_wait_{int(now - t0)}s")
                last_shot = now
        await asyncio.sleep(1)
    raise RuntimeError(f"Hostname not received via ntfy.sh after {HOSTNAME_WAIT_TIMEOUT}s.")

# ── Setup mode ────────────────────────────────────────────────────────────────

def run_setup():
    """
    Launch real Chrome (no automation) to accounts.google.com.
    User signs in normally. Session saved to COLAB_PROFILE.
    """
    COLAB_PROFILE.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Profile: {COLAB_PROFILE}", file=sys.stderr)
    print("[setup] Chrome opening to accounts.google.com.", file=sys.stderr)
    print("[setup] Sign in with your Google account, then close Chrome.", file=sys.stderr)

    proc = subprocess.Popen([
        CHROME_BIN,
        f"--user-data-dir={COLAB_PROFILE}",
        "--no-first-run",
        "--no-default-browser-check",
        "https://accounts.google.com",
    ])
    proc.wait()  # Block until user closes Chrome
    print("[setup] Setup complete. Session saved. Run without --setup to continue.", file=sys.stderr)


# ── Normal run ────────────────────────────────────────────────────────────────

async def trigger_bootstrap(gpu_type: str = "T4 GPU", keep_chrome: bool = False, notebook_url: str | None = None, high_ram: bool = False) -> str:
    if not COLAB_PROFILE.exists():
        raise RuntimeError(
            f"Profile not found at {COLAB_PROFILE}.\n"
            "Run --setup first: python3 scripts/trigger_colab_bootstrap.py --setup"
        )

    _target_url = notebook_url or NOTEBOOK_URL
    chrome_proc = _launch_chrome(_target_url)
    try:
        await _wait_for_cdp()
        print("[trigger] CDP ready. Connecting Playwright...", file=sys.stderr)

        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(CDP_URL)
            ctx = browser.contexts[0] if browser.contexts else await browser.new_context()

            # Auto-accept all JS dialogs (alert/confirm/prompt/beforeunload).
            # "Leave site?" (beforeunload) appears when OAuth tabs close/redirect;
            # accepting it (clicking Leave) allows the navigation to complete.
            async def _accept_dialog(dialog):
                print(f"[trigger] Browser dialog ({dialog.type}): accepting.", file=sys.stderr)
                await dialog.accept()

            def _attach_dialog_handler(pg):
                pg.on("dialog", lambda d: asyncio.create_task(_accept_dialog(d)))

            for pg in ctx.pages:
                _attach_dialog_handler(pg)
            ctx.on("page", _attach_dialog_handler)

            # Find or wait for the Colab page to load
            page = None
            for _ in range(20):
                page = _colab_page(ctx)
                if page and "colab.research.google.com" in page.url:
                    break
                await asyncio.sleep(2)

            if page is None:
                raise RuntimeError("Colab page not found after Chrome launch.")

            # Always navigate to the target URL to pick up the latest code.
            # Chrome restores previous tabs — the tab's cell code may be stale (from
            # before the last sync). Navigating ensures we always run current code.
            print(f"[trigger] Loading notebook from {_target_url[:80]}...", file=sys.stderr)
            await page.goto(_target_url, wait_until="load", timeout=90_000)

            _screenshot_run_init()
            global _RUNTIME_TYPE_SAVE_GPU
            _RUNTIME_TYPE_SAVE_GPU = None
            print(f"[trigger] Colab page: {page.url}", file=sys.stderr)
            # Colab keeps long-lived connections; "networkidle" often never fires.
            await page.wait_for_load_state("load", timeout=90_000)
            await page.wait_for_selector(
                "colab-connect-button, #connect, colab-toolbar-button",
                timeout=120_000,
            )
            print("[trigger] Colab shell ready (toolbar/connect present).", file=sys.stderr)

            page = await _handle_all_modals(ctx, page)
            await _screenshot_process(page, "01_shell_ready")

            # Always set the GPU type first — even if a runtime is live.
            # If a different type is already running, _set_runtime_type_gpu handles
            # the "Disconnect and delete runtime" confirmation, which disconnects
            # the old session so we can connect with the correct type.
            await _set_runtime_type_gpu(page, gpu_type=gpu_type, high_ram=high_ram)
            await asyncio.sleep(2)
            page = _colab_page(ctx) or page
            await _screenshot_process(page, "02_after_set_runtime_type_gpu")

            page = await _ensure_connect_if_idle(
                ctx, page, "After setting runtime type"
            )

            # After potential disconnect from type change, check if still live.
            if await _runtime_is_live(page):
                print("[trigger] Runtime already active with correct type — skipping Connect.", file=sys.stderr)
            else:
                await _click_connect(page)
                await asyncio.sleep(5)

                # Handle modals that appear AFTER clicking Connect (sign-in, not-authored)
                page = _colab_page(ctx) or page
                page = await _handle_all_modals(ctx, page)

                # Retry connect ONLY if button shows fully idle (auth redirect case).
                # Premium GPUs take 30-60s to provision — don't interrupt provisioning.
                btn_state = await page.evaluate("""() => {
                    const host = document.querySelector('colab-connect-button');
                    if (!host || !host.shadowRoot) return 'unknown';
                    const btn = host.shadowRoot.querySelector('button');
                    return btn ? btn.textContent.trim().toLowerCase() : 'unknown';
                }""")
                print(f"[trigger] Connect button state after click: '{btn_state}'", file=sys.stderr)
                if btn_state == "connect":
                    print("[trigger] Connect reverted to idle — retrying once.", file=sys.stderr)
                    await _click_connect(page)
                    await asyncio.sleep(2)

            page = _colab_page(ctx) or page
            await _screenshot_process(page, "03_before_wait_runtime_connected")
            page = await _wait_runtime_connected(ctx)
            page = await _handle_all_modals(ctx, page)
            await _screenshot_process(page, "04_runtime_connected")
            await _assert_runtime_matches_gpu(page, gpu_type)

            # Skip page reload: the notebook was opened from the GitHub URL directly,
            # so content is already correct. Reloading resets Colab Secrets permissions
            # causing userdata.get('NTFY_TOPIC') to fail with SecretNotFoundError.
            page = _colab_page(ctx) or page
            await _screenshot_process(page, "05_pre_run")
            await _assert_runtime_matches_gpu(page, gpu_type)

            start_epoch = int(time.time())
            await page.keyboard.press("Control+F9")
            print("[trigger] Running all cells...", file=sys.stderr)
            await asyncio.sleep(3)
            # "Run anyway" often appears only after Run all (localized UI).
            page = _colab_page(ctx) or page
            page = await _handle_all_modals(ctx, page)
            await _screenshot_process(page, "06_after_run_all_modals")

            # Run Drive OAuth handler concurrently with hostname polling.
            # drive.mount() opens an accounts.google.com/oauth popup; the
            # handler clicks Allow so Drive mounts without user interaction.
            drive_auth_task = asyncio.create_task(_handle_drive_auth_background(ctx))
            try:
                hostname = await _wait_for_hostname_ntfy(ctx, start_epoch)
                # Grace period: drive.mount() runs in the next cell and may open a
                # Google OAuth popup (accounts.google.com/signin/oauth/id) after the
                # hostname is already received. Keep the auth handler alive so it can
                # click Continue without human interaction.
                print("[trigger] Hostname received; waiting for Drive OAuth to complete (60s)...", file=sys.stderr)
                page = _colab_page(ctx) or page
                await _screenshot_process(page, "07_hostname_received")
                await asyncio.sleep(60)
            finally:
                drive_auth_task.cancel()

            page = _colab_page(ctx) or page
            await _screenshot_process(page, "08_before_return_success")
            if keep_chrome:
                print(f"[trigger] Keeping Chrome alive (PID {chrome_proc.pid}) for Colab websocket.", file=sys.stderr)
            return hostname
    finally:
        if not keep_chrome:
            chrome_proc.kill()  # SIGKILL: force-close Chrome without triggering beforeunload ("Leave site?")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Trigger Colab SSH bootstrap")
    parser.add_argument("--setup", action="store_true", help="One-time Google login setup")
    parser.add_argument("--gpu", default="T4 GPU", metavar="TYPE",
                        help="Hardware accelerator label as shown in Colab (default: 'T4 GPU'). "
                             "Examples: 'T4 GPU', 'L4 GPU', 'A100 GPU', 'None'")
    parser.add_argument(
        "--high-ram",
        action="store_true",
        default=False,
        help="Select the High-RAM runtime shape in the Change-runtime-type dialog (if available on the account).",
    )
    parser.add_argument(
        "--debug-dialog",
        action="store_true",
        help="Log md-text-button snapshots when resolving disconnect / runtime-change OK clicks.",
    )
    parser.add_argument(
        "--no-screenshots",
        action="store_true",
        help="Disable progress PNG screenshots (default: write colab_trigger_* under /tmp).",
    )
    parser.add_argument(
        "--screenshot-dir",
        default=None,
        metavar="DIR",
        help="Directory for progress screenshots (default: /tmp).",
    )
    parser.add_argument(
        "--screenshot-interval",
        type=float,
        default=1.0,
        metavar="SEC",
        help="Minimum seconds between automatic viewport screenshots during waits (default: 1).",
    )
    parser.add_argument(
        "--keep-chrome",
        action="store_true",
        help="Keep Chrome running after hostname is returned (maintains Colab websocket for long-running jobs).",
    )
    parser.add_argument(
        "--notebook-url",
        default=None,
        metavar="URL",
        help="Override the Colab notebook URL (default: GitHub-hosted bootstrap notebook). "
             "Use a Drive URL like https://colab.research.google.com/drive/<FILE_ID> to iterate without GitHub commits.",
    )
    parser.add_argument(
        "--cdp-port",
        type=int,
        default=None,
        metavar="PORT",
        help="Chrome remote-debugging port (default: 9222). Override to run a second, "
             "isolated bootstrap session concurrently without killing another session's Chrome.",
    )
    parser.add_argument(
        "--profile-dir",
        default=None,
        metavar="DIR",
        help="Chrome user-data-dir / saved-session profile directory (default: ~/.colab_chrome_profile). "
             "Override together with --cdp-port to run a fully isolated concurrent session. "
             "A new profile dir has no saved Google session yet -- run once with --setup "
             "(passing the same --profile-dir) before normal use.",
    )
    args = parser.parse_args()

    global CDP_PORT, CDP_URL, COLAB_PROFILE
    if args.cdp_port is not None:
        CDP_PORT = args.cdp_port
        CDP_URL = f"http://localhost:{CDP_PORT}"
    if args.profile_dir:
        COLAB_PROFILE = Path(args.profile_dir).expanduser()

    if args.setup:
        run_setup()
        sys.exit(0)

    global _DEBUG_DIALOG, _PROCESS_SCREENSHOTS, _SCREENSHOT_DIR, _SCREENSHOT_INTERVAL_SEC
    _DEBUG_DIALOG = bool(args.debug_dialog)
    _PROCESS_SCREENSHOTS = not bool(args.no_screenshots)
    if args.screenshot_dir:
        _SCREENSHOT_DIR = Path(args.screenshot_dir).expanduser()
    _SCREENSHOT_INTERVAL_SEC = max(0.25, float(args.screenshot_interval))

    try:
        hostname = asyncio.run(trigger_bootstrap(gpu_type=args.gpu, keep_chrome=args.keep_chrome, notebook_url=args.notebook_url, high_ram=args.high_ram))
        print(hostname)
        sys.exit(0)
    except Exception as e:
        print(f"[trigger] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
