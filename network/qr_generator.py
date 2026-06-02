import socket
from pathlib import Path

import qrcode


def get_local_ip():
    """Detect the local IPv4 address for the machine on the LAN."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def generate_qr(url: str, output_path: Path = None):
    """Generate a QR code image for the given URL."""
    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=8,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    return image
