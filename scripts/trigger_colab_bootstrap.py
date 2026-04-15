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
import re
import subprocess
import sys
import time
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


async def _handle_all_modals(ctx, page):
    await _handle_modal(page, "not authored by Google", "Run anyway")
    for text in ["Dismiss", "Got it", "No thanks"]:
        try:
            btn = page.locator(f"text={text}").first
            if await btn.is_visible(timeout=500):
                await btn.click()
        except Exception:
            pass
    return _colab_page(ctx) or page


async def _click_connect(page):
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


async def _wait_for_hostname(ctx):
    print(f"[trigger] Waiting for hostname (up to {HOSTNAME_WAIT_TIMEOUT}s)...", file=sys.stderr)
    deadline = asyncio.get_event_loop().time() + HOSTNAME_WAIT_TIMEOUT
    while asyncio.get_event_loop().time() < deadline:
        page = _colab_page(ctx)
        if page:
            try:
                content = await page.evaluate("""
                    () => {
                        const nodes = document.querySelectorAll(
                            '.output_text, .output_subarea, .cell-output-text, pre, output'
                        );
                        return Array.from(nodes).map(n => n.textContent).join('\\n');
                    }
                """)
                m = HOSTNAME_RE.search(content)
                if m:
                    return m.group(1).strip()
            except Exception:
                pass
        await asyncio.sleep(5)
    raise RuntimeError(f"Hostname not found after {HOSTNAME_WAIT_TIMEOUT}s.")


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

async def trigger_bootstrap() -> str:
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
            await page.wait_for_load_state("networkidle", timeout=60_000)

            page = await _handle_all_modals(ctx, page)
            await _click_connect(page)
            await asyncio.sleep(2)

            # Handle modals that appear AFTER clicking Connect (sign-in, not-authored)
            page = _colab_page(ctx) or page
            page = await _handle_all_modals(ctx, page)

            # Auth redirect may have abandoned the first Connect — retry
            await _click_connect(page)
            await asyncio.sleep(2)

            # T4 GPU picker
            try:
                gpu = page.locator("text=T4 GPU").first
                if await gpu.is_visible(timeout=3_000):
                    await gpu.click()
                    print("[trigger] Selected T4 GPU.", file=sys.stderr)
            except Exception:
                pass

            page = await _wait_runtime_connected(ctx)
            page = await _handle_all_modals(ctx, page)

            await page.keyboard.press("Control+F9")
            print("[trigger] Running all cells...", file=sys.stderr)
            await asyncio.sleep(2)

            hostname = await _wait_for_hostname(ctx)
            await browser.disconnect()
            return hostname
    finally:
        chrome_proc.terminate()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if "--setup" in sys.argv:
        run_setup()
        sys.exit(0)

    try:
        hostname = asyncio.run(trigger_bootstrap())
        print(hostname)
        sys.exit(0)
    except Exception as e:
        print(f"[trigger] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
