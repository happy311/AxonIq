"""Item 9/10 — frontend is CSP-clean: no inline handlers/scripts/styles, no token storage, escaping in place."""
import re
from pathlib import Path

FE = Path(__file__).resolve().parents[1] / "frontend"
HTML = list(FE.glob("*.html"))
JS = list((FE / "js").glob("*.js"))


def test_no_inline_event_handlers_or_scripts_or_style_blocks():
    for f in HTML:
        t = f.read_text()
        assert not re.search(r"\son[a-z]+\s*=\s*[\"']", t, re.I), f"inline handler in {f.name}"
        assert not re.search(r"<script(?![^>]*\ssrc=)[^>]*>", t, re.I), f"inline <script> in {f.name}"
        assert "<style" not in t.lower(), f"<style> block in {f.name}"
        assert "javascript:" not in t.lower()


def test_no_interpolated_handlers_in_js():
    for f in JS:
        t = f.read_text()
        assert not re.search(r"\bon(click|change|input|keydown|submit)\s*=\s*\\?[\"']", t), f"inline handler string in {f.name}"


def test_no_token_in_js_storage():
    for f in JS:
        t = f.read_text()
        assert "localStorage" not in t and "sessionStorage" not in t, f.name
        assert "Authorization" not in t, f"{f.name} builds Authorization headers (cookie auth only)"


def test_escape_function_escapes_quotes():
    t = (FE / "js" / "utils.js").read_text()
    assert "&quot;" in t and "&#39;" in t


def test_admin_js_escapes_and_csv_is_safe():
    t = (FE / "js" / "admin.js").read_text()
    assert "csvCell" in t and "\"'\" + s" in t and "replace(/\"/g, '\"\"')" in t     # formula prefix + RFC-4180 quoting
    assert "${l.action}" not in t and "${e.message}" not in t and "${l.created_at}" not in t
    assert "onclick" not in t


def test_csp_header_has_no_unsafe_inline_for_scripts():
    from api.core.middleware import CSP
    script_src = [d for d in CSP.split("; ") if d.startswith("script-src")][0]
    assert script_src == "script-src 'self'"
    assert "fonts.googleapis.com" in CSP and "fonts.gstatic.com" in CSP
