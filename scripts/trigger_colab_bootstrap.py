#!/usr/bin/env python3
"""
trigger_colab_bootstrap.py

Automatically opens the Colab bootstrap notebook using the existing Chrome
Google session, connects to a GPU runtime, runs all cells, and returns the
cloudflared SSH hostname — with no human involvement.

Usage:
    python3 scripts/trigger_colab_bootstrap.py

Outputs the hostname to stdout on success (e.g. loud-turkey-abc.trycloudflare.com).
Exits with code 1 on failure.

Requirements:
    pip3 install playwright
    playwright install chromium   # or use channel="chrome" for system Chrome
"""

import asyncio
import re
import shutil
import sys
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# ── Configuration ─────────────────────────────────────────────────────────────

NOTEBOOK_URL = (
    "https://colab.research.google.com/github/allyoushawn/recsys_playground"
    "/blob/main/notebooks/ad_hoc/colab_ssh_bootstrap.ipynb"
)

CHROME_PROFILE = (
    Path.home() / "Library/Application Support/Google/Chrome/Default"
)

# How long to wait for GPU runtime allocation (seconds)
RUNTIME_CONNECT_TIMEOUT = 180

# How long to wait for the cloudflared hostname to appear in output (seconds)
HOSTNAME_WAIT_TIMEOUT = 300

# ── Helpers ───────────────────────────────────────────────────────────────────

def _copy_profile_to_tmp() -> Path:
    """
    Copy the Chrome Default profile to a temp dir.
    This avoids the SingletonLock issue when Chrome is already open,
    while preserving all cookies and login state.
    """
    tmp = Path(tempfile.mkdtemp()) / "chrome_profile"
    print("[trigger] Copying Chrome profile to temp dir...", file=sys.stderr)
    shutil.copytree(
        CHROME_PROFILE,
        tmp,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "SingletonLock",
            "SingletonSocket",
            "*.log",
            "*.ldb",
            "*.sst",
            "LOCK",
            "LOG",
            "LOG.old",
        ),
    )
    # Remove any leftover lock files
    for lock in tmp.glob("SingletonLock"):
        lock.unlink(missing_ok=True)
    print("[trigger] Profile copied.", file=sys.stderr)
    return tmp


async def _try_dismiss_dialogs(page):
    """Dismiss any modal dialogs that may appear (e.g. 'Open in Drive' banner)."""
    for selector in [
        "text=Dismiss",
        "text=Got it",
        "text=No thanks",
        "text=Cancel",
        "[aria-label='Close']",
    ]:
        try:
            btn = page.locator(selector).first
            if await btn.is_visible(timeout=1_000):
                await btn.click()
        except Exception:
            pass


async def _connect_runtime(page):
    """
    Click the Connect button and wait for the runtime to become live.
    Tries multiple selector strategies for robustness.
    """
    print("[trigger] Looking for Connect button...", file=sys.stderr)

    # Strategy 1: colab-connect-button element
    try:
        btn = page.locator("colab-connect-button").first
        await btn.click(timeout=10_000)
        print("[trigger] Clicked colab-connect-button.", file=sys.stderr)
    except Exception:
        pass

    # Strategy 2: #connect id
    try:
        btn = page.locator("#connect").first
        if await btn.is_visible(timeout=3_000):
            await btn.click()
            print("[trigger] Clicked #connect.", file=sys.stderr)
    except Exception:
        pass

    # Strategy 3: button with "Connect" text
    try:
        btn = page.get_by_role("button", name=re.compile(r"Connect", re.I)).first
        if await btn.is_visible(timeout=3_000):
            await btn.click()
            print("[trigger] Clicked Connect button by role.", file=sys.stderr)
    except Exception:
        pass

    # If a runtime type picker appeared, pick T4 GPU
    try:
        gpu = page.locator("text=T4 GPU").first
        if await gpu.is_visible(timeout=4_000):
            await gpu.click()
            print("[trigger] Selected T4 GPU runtime.", file=sys.stderr)
    except Exception:
        pass

    # Wait for runtime to become live.
    # Signal: colab-usage-display appears (shows RAM/Disk bars), OR
    #         connect button changes to show usage text.
    print(
        f"[trigger] Waiting for runtime (up to {RUNTIME_CONNECT_TIMEOUT}s)...",
        file=sys.stderr,
    )
    connected = False
    for selector in [
        "colab-usage-display",                    # RAM/Disk usage bar
        "[data-connected='true']",                # connect button attribute
        "text=RAM",                               # usage display text
        "colab-runtime-status[connected]",        # runtime status element
    ]:
        try:
            await page.wait_for_selector(
                selector, timeout=RUNTIME_CONNECT_TIMEOUT * 1000
            )
            print(f"[trigger] Runtime live (detected via '{selector}').", file=sys.stderr)
            connected = True
            break
        except PlaywrightTimeout:
            continue

    if not connected:
        raise RuntimeError(
            "Runtime did not connect within timeout. "
            "Check that a GPU runtime is available on this account."
        )


async def _run_all_cells(page):
    """Trigger Run All via keyboard shortcut."""
    print("[trigger] Running all cells (Ctrl+F9)...", file=sys.stderr)
    await page.keyboard.press("Control+F9")
    # Small pause to ensure execution starts
    await asyncio.sleep(2)


async def _wait_for_hostname(page) -> str:
    """
    Wait for the trycloudflare.com hostname to appear in any cell output.
    Returns the hostname string.
    """
    print(
        f"[trigger] Waiting for cloudflared hostname (up to {HOSTNAME_WAIT_TIMEOUT}s)...",
        file=sys.stderr,
    )

    # Poll the page DOM every 5 seconds for the hostname pattern
    deadline = asyncio.get_event_loop().time() + HOSTNAME_WAIT_TIMEOUT
    hostname_pattern = re.compile(r'([\w-]+\.trycloudflare\.com)')

    while asyncio.get_event_loop().time() < deadline:
        try:
            # Get all text content from output cells
            content = await page.evaluate("""
                () => {
                    const outputs = document.querySelectorAll(
                        '.output_text, .output_subarea, .cell-output-text, pre'
                    );
                    return Array.from(outputs).map(el => el.textContent).join('\\n');
                }
            """)
            match = hostname_pattern.search(content)
            if match:
                return match.group(1).strip()
        except Exception:
            pass

        # Also try waiting for the text directly via Playwright
        try:
            el = await page.wait_for_selector(
                "text=/trycloudflare\\.com/",
                timeout=5_000,
            )
            text = await el.text_content()
            match = hostname_pattern.search(text)
            if match:
                return match.group(1).strip()
        except PlaywrightTimeout:
            pass

        await asyncio.sleep(5)

    raise RuntimeError(
        f"Hostname not found in Colab output after {HOSTNAME_WAIT_TIMEOUT}s. "
        "Check that Cell 3 ran successfully."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

async def trigger_bootstrap() -> str:
    tmp_profile = _copy_profile_to_tmp()

    try:
        async with async_playwright() as p:
            print("[trigger] Launching Chrome with existing Google session...", file=sys.stderr)
            ctx = await p.chromium.launch_persistent_context(
                str(tmp_profile),
                headless=False,          # keep visible: handles any auth/2FA prompts
                channel="chrome",        # use system Chrome (has Google cookies)
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-first-run",
                    "--no-default-browser-check",
                ],
                ignore_https_errors=True,
            )

            page = await ctx.new_page()

            print(f"[trigger] Opening: {NOTEBOOK_URL}", file=sys.stderr)
            await page.goto(NOTEBOOK_URL, wait_until="domcontentloaded", timeout=60_000)
            await asyncio.sleep(3)  # let dynamic content settle

            await _try_dismiss_dialogs(page)
            await _connect_runtime(page)
            await _try_dismiss_dialogs(page)
            await _run_all_cells(page)
            hostname = await _wait_for_hostname(page)

            await ctx.close()
            return hostname

    finally:
        shutil.rmtree(tmp_profile.parent, ignore_errors=True)


def main():
    try:
        hostname = asyncio.run(trigger_bootstrap())
        print(hostname)  # stdout — for shell capture: HOSTNAME=$(python3 ...)
        sys.exit(0)
    except Exception as e:
        print(f"[trigger] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
