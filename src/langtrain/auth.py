"""
auth.py — Browser-based authentication for Langtrain CLI

Flow (identical to GitHub CLI / Claude Code):
  1. CLI generates a random one-time state token
  2. CLI starts a local HTTP server on a free port (e.g. localhost:54321)
  3. CLI opens (or prints) the URL:
       https://app.langtrain.xyz/cli-auth?state=<state>&port=<port>
  4. User logs in / signs up in the browser
  5. Browser redirects to http://localhost:<port>/callback?token=<api_key>&state=<state>
  6. CLI receives the token, verifies state, saves ~/.langtrain/credentials.json
  7. CLI prints "✓ Authenticated as user@example.com"

No API key copy-pasting required.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rich.console import Console

console = Console(highlight=False)

# Where credentials live
CREDS_PATH = Path.home() / ".langtrain" / "credentials.json"
BASE_URL    = os.environ.get("LANGTRAIN_BASE_URL", "https://api.langtrain.xyz").rstrip("/")
APP_URL     = os.environ.get("LANGTRAIN_APP_URL",  "https://app.langtrain.xyz")


# ─────────────────────────────────────────────────────────────────────────────
# Local callback server
# ─────────────────────────────────────────────────────────────────────────────

class _CallbackHandler(BaseHTTPRequestHandler):
    """Handles the single redirect from the browser after the user authenticates."""

    result: dict | None = None   # shared across threads via the server instance

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/callback":
            self._respond(404, "Not found")
            return

        params = parse_qs(parsed.query)
        token  = (params.get("token") or params.get("api_key") or [""])[0]
        state  = (params.get("state") or [""])[0]
        error  = (params.get("error") or [""])[0]

        if error:
            self._respond(200, _html_page(
                title="Authentication failed",
                body=f"<p style='color:red'>{error}</p><p>Return to your terminal.</p>",
            ))
            self.server.auth_result = {"error": error}
            return

        if not token:
            self._respond(400, "Missing token")
            return

        # Return a nice success page the user sees in the browser
        self._respond(200, _html_page(
            title="Authenticated!",
            body="<p>You're logged in to Langtrain. You can close this tab and return to your terminal.</p>",
        ))
        self.server.auth_result = {"token": token, "state": state}

    def _respond(self, code: int, body: str):
        encoded = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_):
        pass   # silence default access log


def _html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 480px; margin: 80px auto;
          padding: 0 24px; color: #111; }}
  h1   {{ font-size: 1.4rem; }}
  p    {{ color: #555; }}
  .logo {{ font-weight: 700; font-size: 1.8rem; letter-spacing: -1px; margin-bottom: 8px; }}
</style>
</head><body>
<div class="logo">⚡ Langtrain</div>
<h1>{title}</h1>
{body}
</body></html>"""


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ─────────────────────────────────────────────────────────────────────────────
# Main login flow
# ─────────────────────────────────────────────────────────────────────────────

def browser_login(timeout: int = 120) -> None:
    """
    Open the browser for one-click authentication.

    Saves the API token to ~/.langtrain/credentials.json on success.
    Raises SystemExit on failure or timeout.
    """
    state = secrets.token_urlsafe(24)
    port  = _find_free_port()

    # Build the auth URL — the web app will redirect to our local server
    auth_url = f"{APP_URL}/cli-auth?state={state}&port={port}&source=cli"

    # Start local HTTP server in a background thread
    server = HTTPServer(("localhost", port), _CallbackHandler)
    server.auth_result = None  # type: ignore[attr-defined]
    server.timeout     = 2     # poll every 2s so we can check timeout

    def _serve():
        deadline = time.time() + timeout
        while server.auth_result is None and time.time() < deadline:
            server.handle_request()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    # Open browser
    console.print()
    console.print("[bold]⚡ Langtrain Login[/]")
    console.print()

    opened = webbrowser.open(auth_url)
    if opened:
        console.print(f"  Opening your browser…")
        console.print(f"  [dim]If the browser didn't open, visit:[/]")
        console.print(f"  [cyan underline]{auth_url}[/]")
    else:
        console.print(f"  [yellow]Could not open browser automatically.[/]")
        console.print(f"  Open this URL to authenticate:")
        console.print()
        console.print(f"  [bold cyan]{auth_url}[/]")

    console.print()
    console.print(f"  [dim]Waiting for authentication…  (Ctrl+C to cancel)[/]")
    console.print()

    # Wait for the callback
    deadline = time.time() + timeout
    try:
        while server.auth_result is None:  # type: ignore[attr-defined]
            if time.time() > deadline:
                console.print("[red]✖  Timed out waiting for authentication.[/]")
                console.print("[dim]  Run 'lt login' to try again.[/]")
                sys.exit(1)
            time.sleep(0.25)
    except KeyboardInterrupt:
        console.print("\n[dim]  Login cancelled.[/]")
        sys.exit(0)

    result = server.auth_result  # type: ignore[attr-defined]

    if result.get("error"):
        console.print(f"[red]✖  Authentication failed: {result['error']}[/]")
        sys.exit(1)

    if result.get("state") != state:
        console.print("[red]✖  Security error: state mismatch. Please try again.[/]")
        sys.exit(1)

    token = result["token"]

    # Verify and fetch user info
    try:
        import requests as _req
        r = _req.get(
            f"{BASE_URL}/v1/users/me",
            headers={"x-api-key": token},
            timeout=8,
        )
        user = r.json() if r.ok else {}
    except Exception:
        user = {}

    # Save credentials
    CREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CREDS_PATH.write_text(json.dumps({
        "api_key": token,
        "email":   user.get("email", ""),
        "plan":    user.get("plan", ""),
    }, indent=2))
    CREDS_PATH.chmod(0o600)   # user-readable only

    email = user.get("email", "")
    plan  = user.get("plan", "free").title()

    console.print(f"[bold green]✔  Authenticated![/]")
    if email:
        console.print(f"   Email: {email}")
    console.print(f"   Plan:  {plan}")
    console.print()
    console.print(f"  [dim]Run [bold]lt[/bold] to start fine-tuning.[/]")
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Credential helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_credentials() -> dict:
    """Return saved credentials or empty dict."""
    try:
        return json.loads(CREDS_PATH.read_text())
    except Exception:
        return {}


def get_api_key() -> str:
    """Return API key from credentials file or LANGTRAIN_API_KEY env var."""
    return (
        os.environ.get("LANGTRAIN_API_KEY")
        or load_credentials().get("api_key")
        or ""
    )


def is_authenticated() -> bool:
    return bool(get_api_key())
