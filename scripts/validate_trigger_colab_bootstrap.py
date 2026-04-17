#!/usr/bin/env python3
"""
Automated validation for trigger_colab_bootstrap.py (no real Colab / GPU).

Loads the trigger module by path and checks:
  - Trusted-notebook modal dismissal (en + zh-TW + regex-only)
  - Hostname extraction from shadow DOM + iframe

Run from repo root:
    python3 scripts/validate_trigger_colab_bootstrap.py

Exit 0 if all checks pass.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

from playwright.async_api import async_playwright


def _load_trigger_module():
    root = Path(__file__).resolve().parent.parent
    path = root / "scripts" / "trigger_colab_bootstrap.py"
    spec = importlib.util.spec_from_file_location("trigger_colab_bootstrap", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


async def _test_zh_tw_trusted_modal(tcb) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(
            """
            <body>
              <p>警告：這個筆記本並非由 Google 編寫</p>
              <p>從「GitHub」載入</p>
              <button type="button">取消</button>
              <button type="button">仍要執行</button>
            </body>
            """
        )
        ok = await tcb._dismiss_trusted_notebook_warning(page)
        assert ok, "zh-TW trusted modal should be dismissed"
        await browser.close()


async def _test_en_trusted_modal(tcb) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(
            """
            <body>
              <p>This notebook was not authored by Google.</p>
              <button>Cancel</button>
              <button>Run anyway</button>
            </body>
            """
        )
        ok = await tcb._dismiss_trusted_notebook_warning(page)
        assert ok, "English trusted modal should be dismissed"
        await browser.close()


async def _test_regex_only_trusted_modal(tcb) -> None:
    """No matching title substring — only the primary-action button (es)."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(
            """
            <body>
              <p>Algo en español sin Google en el título</p>
              <button>Ejecutar de todos modos</button>
            </body>
            """
        )
        ok = await tcb._dismiss_trusted_notebook_warning(page)
        assert ok, "Regex fallback should click Spanish Run-anyway label"
        await browser.close()


async def _test_hostname_shadow_dom(tcb) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(
            """
            <body>
              <div id="host"></div>
              <script>
                const el = document.getElementById('host');
                const sr = el.attachShadow({mode: 'open'});
                sr.innerHTML = '<pre>ssh loud-turkey-abc.trycloudflare.com</pre>';
              </script>
            </body>
            """
        )
        host = await tcb._hostname_from_frame(page.main_frame)
        assert host == "loud-turkey-abc.trycloudflare.com", host
        await browser.close()


async def _test_hostname_iframe(tcb) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_content(
            """
            <iframe id="f" srcdoc="<pre>host-x99.trycloudflare.com</pre>"></iframe>
            """
        )
        await page.wait_for_selector("iframe#f")
        found = None
        for frame in page.frames:
            h = await tcb._hostname_from_frame(frame)
            if h:
                found = h
                break
        assert found == "host-x99.trycloudflare.com", found
        await browser.close()


async def main() -> int:
    tcb = _load_trigger_module()
    # Sync checks (no browser): accelerator token parsing for GPU verify
    assert tcb._gpu_match_keyword("T4 GPU") == "T4"
    assert tcb._gpu_match_keyword("  L4 GPU ") == "L4"
    assert tcb._gpu_match_keyword("None") is None
    print("[validate] gpu keyword mapping OK", flush=True)

    tests = [
        ("en trusted modal", _test_en_trusted_modal(tcb)),
        ("zh-TW trusted modal", _test_zh_tw_trusted_modal(tcb)),
        ("regex-only trusted modal (es)", _test_regex_only_trusted_modal(tcb)),
        ("hostname in shadow DOM", _test_hostname_shadow_dom(tcb)),
        ("hostname in iframe", _test_hostname_iframe(tcb)),
    ]
    for name, coro in tests:
        print(f"[validate] {name}...", flush=True)
        await coro
        print(f"[validate] {name} OK", flush=True)
    print("[validate] All automated checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except AssertionError as e:
        print(f"[validate] FAIL: {e}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    except Exception as e:
        print(f"[validate] ERROR: {e}", file=sys.stderr, flush=True)
        raise SystemExit(2)
