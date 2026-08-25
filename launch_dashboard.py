"""Start the Streamlit server, wait until it is really listening, then open the browser.

Why this file exists: `open http://localhost:8501` two seconds after launch
often hits the browser before the server is ready, so Safari/Chrome says
"site can't be reached." We also bind to 127.0.0.1 (IPv4) so `localhost`
does not resolve to IPv6 while the server is only on IPv4.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8501
URL = f"http://{HOST}:{PORT}"
ROOT = Path(__file__).resolve().parent


def port_is_open(host: str = HOST, port: int = PORT) -> bool:
    """True when something on this machine accepts connections on host:port."""
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def main() -> int:
    python = sys.executable
    app = ROOT / "app.py"
    if not app.exists():
        print(f"Could not find {app}")
        return 1

    if port_is_open():
        print(f"Something is already running at {URL}")
        print("Opening it in your browser. If that tab is blank, quit the old")
        print("server (Ctrl+C in the other terminal) and run:  make app")
        webbrowser.open(URL)
        return 0

    cmd = [
        python,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        HOST,
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print("Starting the dashboard server …")
    print("Keep this terminal open. Closing it turns the site off.")
    print(f"When ready, the app is:  {URL}")
    print()

    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    deadline = time.time() + 45
    while time.time() < deadline:
        if proc.poll() is not None:
            print()
            print("The server quit before it started listening.")
            print("Scroll up for a Python error (missing package, bad import).")
            print("If you have not run setup yet:  make setup")
            return proc.returncode or 1
        if port_is_open():
            print()
            print(f"Server is up. Opening {URL}")
            webbrowser.open(URL)
            break
        time.sleep(0.4)
    else:
        print()
        print("Timed out waiting for the server. Check this terminal for errors.")
        print(f"You can still try {URL} yourself.")
        proc.terminate()
        return 1

    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
