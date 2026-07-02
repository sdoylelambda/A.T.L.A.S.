#!/usr/bin/env python3
"""
api/gen_qr.py

Generates a QR code from the Atlas API key and displays it on screen.
Scan with the Atlas mobile app to auto-fill the API key in Settings.

Usage:
    python3 ~/dev/A.T.L.A.S./api/gen_qr.py

Requires:
    pip install qrcode --break-system-packages
"""

import os
import sys
import subprocess
from pathlib import Path

API_KEY_PATH = Path.home() / ".config/atlas/api_key"
QR_OUTPUT    = Path("/tmp/atlas_api_key_qr.png")


def main():
    if not API_KEY_PATH.exists():
        print(f"[gen_qr] API key not found at {API_KEY_PATH}")
        print("[gen_qr] Generate one with: openssl rand -hex 32 > ~/.config/atlas/api_key")
        sys.exit(1)

    api_key = API_KEY_PATH.read_text().strip()
    if not api_key:
        print("[gen_qr] API key file is empty.")
        sys.exit(1)

    try:
        import qrcode
    except ImportError:
        print("[gen_qr] qrcode not installed.")
        print("[gen_qr] Run: pip install qrcode --break-system-packages")
        sys.exit(1)

    print(f"[gen_qr] Generating QR code for API key...")
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(api_key)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(QR_OUTPUT)

    print(f"[gen_qr] QR saved to {QR_OUTPUT}")
    print(f"[gen_qr] Opening image — scan with Atlas mobile app...")
    print(f"[gen_qr] Close the image when done.")

    # Open with whatever viewer is available
    for viewer in ["eog", "feh", "xdg-open", "display"]:
        if subprocess.run(["which", viewer], capture_output=True).returncode == 0:
            subprocess.Popen([viewer, str(QR_OUTPUT)])
            break
    else:
        print(f"[gen_qr] No image viewer found — open manually: {QR_OUTPUT}")

    print("[gen_qr] Done. The QR code encodes only your API key — keep it private.")


if __name__ == "__main__":
    main()
