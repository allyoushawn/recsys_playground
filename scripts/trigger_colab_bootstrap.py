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

Requirements:
    pip3 install playwright
"""

import asyncio
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# ── Configuration ─────────────────────────────────────────────────────────────

NOTEBOOK_URL = (
    "https://colab.research.google.com/github/allyoushawn/recsys_playground"
    "/blob/main/notebooks/ad_hoc/colab_ssh_bootstrap.ipynb"
)

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
COLAB_PROFILE = Path.home() / ".colab_chrome_profile"
CDP_PORT = 9222
CDP_URL = f"http://localhost:{CDP_PORT}"

RUNTIME_CONNECT_TIMEOUT = 240
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


async def _handle_all_modals(ctx, page):
    await _dismiss_trusted_notebook_warning(page)
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


async def _set_runtime_type_gpu(page, gpu_type: str = "T4 GPU") -> bool:
    """Open the Connect-button dropdown → Change runtime type → select radio → Save.

    The hardware accelerator is a set of radio buttons (CPU, T4 GPU, L4 GPU, …).
    Clicking the radio label selects it; then Save commits the choice.

    Returns True if the runtime type was set, False otherwise.
    """
    try:
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

        # ── Step 3: select the GPU radio button ───────────────────────────────
        # The dialog radio buttons are accessible by name via the accessibility tree.
        radio = page.get_by_role("radio", name=gpu_type)
        try:
            await radio.click(timeout=5_000)
            print(f"[trigger] Selected radio: {gpu_type}.", file=sys.stderr)
        except Exception:
            await page.keyboard.press("Escape")
            print(f"[trigger] '{gpu_type}' radio not found — check account tier.", file=sys.stderr)
            return False
        await asyncio.sleep(0.5)

        # ── Step 3b: handle "Disconnect and delete runtime" confirmation ───────
        # When an existing runtime is active and the GPU type changes, Colab
        # overlays a confirmation before showing Save. Click OK to proceed.
        # Poll for up to 8s since the dialog appears asynchronously.
        confirmed = False
        deadline = asyncio.get_event_loop().time() + 8
        while asyncio.get_event_loop().time() < deadline:
            clicked = await page.evaluate("""() => {
                // Recursively search shadow DOMs for a visible button with text 'OK'
                function findOK(root) {
                    for (const el of root.querySelectorAll('*')) {
                        if (el.shadowRoot) {
                            const found = findOK(el.shadowRoot);
                            if (found) return found;
                        }
                        if ((el.tagName === 'BUTTON' || el.getAttribute('role') === 'button') &&
                            el.textContent.trim() === 'OK') {
                            const r = el.getBoundingClientRect();
                            if (r.width > 0 && r.height > 0) return el;
                        }
                    }
                    return null;
                }
                const btn = findOK(document);
                if (btn) { btn.click(); return true; }
                return false;
            }""")
            if clicked:
                print("[trigger] Confirmed 'Disconnect and delete runtime' → OK.", file=sys.stderr)
                await asyncio.sleep(1.5)
                confirmed = True
                break
            await asyncio.sleep(0.5)
        if not confirmed:
            print("[trigger] No disconnect confirmation dialog — continuing.", file=sys.stderr)

        # ── Step 4: click Save ────────────────────────────────────────────────
        save_btn = page.get_by_role("button", name="Save")
        try:
            await save_btn.click(timeout=5_000)
            print(f"[trigger] Runtime type saved ({gpu_type}).", file=sys.stderr)
        except Exception:
            await page.keyboard.press("Escape")
            print("[trigger] Save button not found.", file=sys.stderr)
            return False
        await asyncio.sleep(1.5)
        return True

    except Exception as exc:
        print(f"[trigger] _set_runtime_type_gpu error: {exc}", file=sys.stderr)
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass
        return False


async def _runtime_is_live(page) -> bool:
    """Return True if the page already shows an active or resuming runtime."""
    live_selectors = [
        "colab-usage-display", "text=RAM",
        "[data-connected='true']", "colab-runtime-status",
    ]
    resuming_texts = ["Resuming session", "Connecting", "Initializing"]
    for sel in live_selectors:
        try:
            el = await page.query_selector(sel)
            if el and await el.is_visible():
                return True
        except Exception:
            pass
    for text in resuming_texts:
        try:
            el = await page.query_selector(f"text={text}")
            if el and await el.is_visible():
                return True
        except Exception:
            pass
    # Also check the connect button's shadow DOM text — "Connecting" appears
    # inside the shadow root when a GPU runtime is provisioning.
    try:
        btn_text = await page.evaluate("""() => {
            const host = document.querySelector('colab-connect-button');
            if (!host || !host.shadowRoot) return '';
            const btn = host.shadowRoot.querySelector('button');
            return btn ? btn.textContent.trim().toLowerCase() : '';
        }""")
        if btn_text and btn_text != "connect":
            return True  # e.g. "connecting", "initializing", "reconnecting"
    except Exception:
        pass
    return False


async def _wait_runtime_connected(ctx):
    print(f"[trigger] Waiting for runtime (up to {RUNTIME_CONNECT_TIMEOUT}s)...", file=sys.stderr)
    connected_selectors = [
        "colab-usage-display", "text=RAM",
        "[data-connected='true']", "colab-runtime-status",
    ]
    deadline = asyncio.get_event_loop().time() + RUNTIME_CONNECT_TIMEOUT
    while asyncio.get_event_loop().time() < deadline:
        page = _colab_page(ctx)
        if page:
            for sel in connected_selectors:
                try:
                    el = await page.query_selector(sel)
                    if el and await el.is_visible():
                        print(f"[trigger] Runtime live (via '{sel}').", file=sys.stderr)
                        return page
                except Exception:
                    pass
        await asyncio.sleep(3)
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


async def _wait_for_hostname_ntfy(start_epoch: int) -> str:
    """Poll ntfy.sh for the hostname POSTed by the bootstrap notebook."""
    import json
    import urllib.request
    print(f"[trigger] Polling ntfy.sh for hostname (up to {HOSTNAME_WAIT_TIMEOUT}s)...", file=sys.stderr)
    deadline = time.time() + HOSTNAME_WAIT_TIMEOUT
    while time.time() < deadline:
        try:
            url = f"https://ntfy.sh/{NTFY_TOPIC}/json?since={start_epoch}&poll=1"
            with urllib.request.urlopen(url, timeout=10) as resp:
                lines = resp.read().decode().strip().splitlines()
            for line in reversed(lines):
                try:
                    msg = json.loads(line)
                    hostname = msg.get("message", "").strip()
                    if HOSTNAME_RE.match(hostname):
                        print(f"[trigger] Hostname received: {hostname}", file=sys.stderr)
                        return hostname
                except Exception:
                    pass
        except Exception as e:
            print(f"[trigger] ntfy.sh poll error: {e}", file=sys.stderr)
        await asyncio.sleep(NTFY_POLL_INTERVAL)
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

async def trigger_bootstrap(gpu_type: str = "T4 GPU") -> str:
    if not COLAB_PROFILE.exists():
        raise RuntimeError(
            f"Profile not found at {COLAB_PROFILE}.\n"
            "Run --setup first: python3 scripts/trigger_colab_bootstrap.py --setup"
        )

    chrome_proc = _launch_chrome(NOTEBOOK_URL)
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

            print(f"[trigger] Colab page: {page.url}", file=sys.stderr)
            # Colab keeps long-lived connections; "networkidle" often never fires.
            await page.wait_for_load_state("load", timeout=90_000)
            await page.wait_for_selector(
                "colab-connect-button, #connect, colab-toolbar-button",
                timeout=120_000,
            )
            print("[trigger] Colab shell ready (toolbar/connect present).", file=sys.stderr)

            page = await _handle_all_modals(ctx, page)

            # Set GPU runtime type BEFORE connecting (only when no runtime is active).
            # If a runtime is already live, changing type would restart it — skip.
            if await _runtime_is_live(page):
                print("[trigger] Runtime already active or resuming — skipping Connect click.", file=sys.stderr)
            else:
                await _set_runtime_type_gpu(page, gpu_type=gpu_type)
                # Give Colab a moment to settle after the runtime type dialog closes
                await asyncio.sleep(2)

                await _click_connect(page)
                await asyncio.sleep(5)

                # Handle modals that appear AFTER clicking Connect (sign-in, not-authored)
                page = _colab_page(ctx) or page
                page = await _handle_all_modals(ctx, page)

                # Retry connect ONLY if runtime shows no sign of activity at all
                # (e.g. auth redirect). Premium GPUs (L4, A100) take 30-60s to
                # provision — do NOT retry during provisioning or it cancels the request.
                btn_state = await page.evaluate("""() => {
                    const host = document.querySelector('colab-connect-button');
                    if (!host || !host.shadowRoot) return 'unknown';
                    const btn = host.shadowRoot.querySelector('button');
                    return btn ? btn.textContent.trim().toLowerCase() : 'unknown';
                }""")
                print(f"[trigger] Connect button state after click: '{btn_state}'", file=sys.stderr)
                if btn_state == "connect":
                    # Button reverted to idle — auth redirect likely ate the click
                    print("[trigger] Connect reverted to idle — retrying once.", file=sys.stderr)
                    await _click_connect(page)
                    await asyncio.sleep(2)

            page = await _wait_runtime_connected(ctx)
            page = await _handle_all_modals(ctx, page)

            # Reload page to force Colab to fetch latest notebook content from GitHub.
            # Without this, Colab may silently show a Drive-autosaved version (which
            # can be stale if the notebook was edited locally and pushed to GitHub).
            print("[trigger] Reloading page to fetch latest notebook content from GitHub...", file=sys.stderr)
            await page.reload(wait_until="load")
            await page.wait_for_selector(
                "colab-connect-button, #connect, colab-toolbar-button",
                timeout=60_000,
            )
            page = await _handle_all_modals(ctx, page)
            # Runtime reconnects automatically after reload; wait for it to be live again.
            page = await _wait_runtime_connected(ctx)
            page = await _handle_all_modals(ctx, page)

            start_epoch = int(time.time())
            await page.keyboard.press("Control+F9")
            print("[trigger] Running all cells...", file=sys.stderr)
            await asyncio.sleep(3)
            # "Run anyway" often appears only after Run all (localized UI).
            page = _colab_page(ctx) or page
            page = await _handle_all_modals(ctx, page)

            # Run Drive OAuth handler concurrently with hostname polling.
            # drive.mount() opens an accounts.google.com/oauth popup; the
            # handler clicks Allow so Drive mounts without user interaction.
            drive_auth_task = asyncio.create_task(_handle_drive_auth_background(ctx))
            try:
                hostname = await _wait_for_hostname_ntfy(start_epoch)
                # Grace period: drive.mount() runs in the next cell and may open a
                # Google OAuth popup (accounts.google.com/signin/oauth/id) after the
                # hostname is already received. Keep the auth handler alive so it can
                # click Continue without human interaction.
                print("[trigger] Hostname received; waiting for Drive OAuth to complete (60s)...", file=sys.stderr)
                await asyncio.sleep(60)
            finally:
                drive_auth_task.cancel()

            return hostname
    finally:
        chrome_proc.kill()  # SIGKILL: force-close Chrome without triggering beforeunload ("Leave site?")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Trigger Colab SSH bootstrap")
    parser.add_argument("--setup", action="store_true", help="One-time Google login setup")
    parser.add_argument("--gpu", default="T4 GPU", metavar="TYPE",
                        help="Hardware accelerator label as shown in Colab (default: 'T4 GPU'). "
                             "Examples: 'T4 GPU', 'L4 GPU', 'A100 GPU', 'None'")
    args = parser.parse_args()

    if args.setup:
        run_setup()
        sys.exit(0)

    try:
        hostname = asyncio.run(trigger_bootstrap(gpu_type=args.gpu))
        print(hostname)
        sys.exit(0)
    except Exception as e:
        print(f"[trigger] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
