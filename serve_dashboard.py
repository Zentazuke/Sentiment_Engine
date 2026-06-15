"""serve_dashboard.py - serve the built React dashboard + proxy /api to the engine.

Mirrors the bot's dashboard_server.py pattern: a tiny stdlib HTTP server, no
extra dependencies, no Node needed on the server. It serves the static build
from ./dashboard_dist and reverse-proxies every /api/* request to the engine on
127.0.0.1:8787 (stripping the /api prefix), so the dashboard's relative calls
work unchanged in production.

Usage:
    python serve_dashboard.py              # default port 8788
    python serve_dashboard.py 9000         # custom port

Build the dashboard once on your PC (where npm works):
    cd sentiment_dashboard_react/sentiment_dashboard_react
    npm run build                          # produces dist/
then copy dist/* into this repo's dashboard_dist/ on the server, e.g.:
    scp -r dist/* kronos@SERVER:~/Sentiment_Engine/dashboard_dist/

Reach it over Tailscale at http://<server-tailscale-ip>:8788 (same as the bot
dashboard on 8765). The engine API itself stays bound to 127.0.0.1 and is never
exposed; only this read-mostly dashboard port is reachable, and only over your
private Tailscale network / behind ufw.
"""

from __future__ import annotations

import os
import socket
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = Path(os.getenv("SENTIMENT_DASHBOARD_DIST", BASE_DIR / "dashboard_dist"))
ENGINE_URL = os.getenv("SENTIMENT_ENGINE_URL", "http://127.0.0.1:8787").rstrip("/")
DEFAULT_PORT = int(os.getenv("SENTIMENT_DASHBOARD_PORT", "8788"))

_CTYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".map": "application/json",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "SentimentDash/1.0"

    def log_message(self, *args):  # quiet; systemd journal would flood otherwise
        pass

    # ---- /api/* -> engine (strip the /api prefix) ----
    def _proxy(self, method: str) -> None:
        upstream = ENGINE_URL + self.path[len("/api"):]
        length = int(self.headers.get("Content-Length", 0) or 0)
        body = self.rfile.read(length) if length else None
        req = urllib.request.Request(upstream, data=body, method=method)
        ctype = self.headers.get("Content-Type")
        if ctype:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as exc:  # forward the engine's error status/body
            data = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:  # engine down/unreachable
            msg = f'{{"error":"engine unreachable: {type(exc).__name__}"}}'.encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)

    # ---- static files with SPA fallback to index.html ----
    def _serve_static(self) -> None:
        rel = self.path.split("?", 1)[0].lstrip("/")
        target = (DIST_DIR / rel).resolve()
        # path-traversal guard
        if not str(target).startswith(str(DIST_DIR.resolve())):
            self.send_error(403)
            return
        if not target.is_file():
            target = DIST_DIR / "index.html"  # SPA fallback
        if not target.is_file():
            self.send_error(404, "dashboard build not found; run npm run build and copy dist/")
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _CTYPES.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            self._proxy("GET")
        else:
            self._serve_static()

    def do_POST(self) -> None:
        if self.path.startswith("/api/"):
            self._proxy("POST")
        else:
            self.send_error(404)


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    if not DIST_DIR.is_file() and not (DIST_DIR / "index.html").is_file():
        print(f"[warn] no build at {DIST_DIR} - run 'npm run build' and copy dist/* there.")
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"Sentiment dashboard serving {DIST_DIR}")
    print(f"  proxying /api -> {ENGINE_URL}")
    print(f"  local:     http://127.0.0.1:{port}")
    print(f"  this box:  http://{_lan_ip()}:{port}  (use the Tailscale IP from another device)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
