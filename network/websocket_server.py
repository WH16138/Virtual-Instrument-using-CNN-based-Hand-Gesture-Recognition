import asyncio
import cv2
import logging
import numpy as np
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from websockets import serve
from websockets.exceptions import ConnectionClosedOK

from .frame_receiver import FrameReceiver
from .qr_generator import generate_qr, get_local_ip

LOG = logging.getLogger("network.websocket_server")

DEFAULT_HTTP_PORT = 8000
DEFAULT_WS_PORT = 8765


class WebSocketFrameServer:
    def __init__(self, receiver: FrameReceiver, http_port: int = DEFAULT_HTTP_PORT, ws_port: int = DEFAULT_WS_PORT):
        self.receiver = receiver
        self.http_port = http_port
        self.ws_port = ws_port
        self.web_root = Path(__file__).resolve().parent.parent / "web"
        self._http_server = None
        self._http_thread = None
        self._ws_server = None
        self._ws_thread = None
        self._ws_loop = None
        self._ws_stop_event = None
        self.page_url = None
        self.qr_code_path = None

    def start(self):
        try:
            self._start_http_server()
            self._start_ws_server()
            self._print_qr_code()
        except Exception as exc:
            LOG.error("Failed to start camera streaming server: %s", exc)
            self.stop()
            raise

    def _start_http_server(self):
        handler = lambda *args, directory=str(self.web_root), **kwargs: SimpleHTTPRequestHandler(*args, directory=directory, **kwargs)
        self._http_server = ThreadingHTTPServer(("0.0.0.0", self.http_port), handler)
        self._http_thread = threading.Thread(target=self._http_server.serve_forever, daemon=True)
        self._http_thread.start()
        LOG.info("Static web server listening on http://0.0.0.0:%d", self.http_port)

    async def _ws_handler(self, websocket):
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        LOG.info("Mobile client connected: %s", client_ip)
        self.receiver.set_client_connected(True)

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    frame = self._decode_jpeg(message)
                    if frame is not None:
                        self.receiver.update_frame(frame)
                else:
                    LOG.debug("Ignoring non-binary websocket message")
        except ConnectionClosedOK:
            LOG.info("Mobile client disconnected cleanly: %s", client_ip)
        except Exception as exc:
            LOG.warning("WebSocket error: %s", exc)
        finally:
            self.receiver.set_client_connected(False)
            LOG.info("Mobile client disconnected: %s", client_ip)

    def _start_ws_server(self):
        startup_ready = threading.Event()
        startup_error = []

        async def serve_ws():
            self._ws_loop = asyncio.get_running_loop()
            self._ws_stop_event = asyncio.Event()
            try:
                async with serve(self._ws_handler, "0.0.0.0", self.ws_port):
                    LOG.info("WebSocket server listening on ws://0.0.0.0:%d", self.ws_port)
                    startup_ready.set()
                    await self._ws_stop_event.wait()
            except Exception as exc:
                startup_error.append(exc)
                startup_ready.set()

        self._ws_thread = threading.Thread(target=lambda: asyncio.run(serve_ws()), daemon=True)
        self._ws_thread.start()
        startup_ready.wait(timeout=3)

        if startup_error:
            raise startup_error[0]

        if not startup_ready.is_set():
            raise TimeoutError(f"Timed out while starting WebSocket server on port {self.ws_port}")

    def _print_qr_code(self):
        local_ip = get_local_ip()
        self.page_url = f"http://{local_ip}:{self.http_port}/?ws_port={self.ws_port}"
        self.qr_code_path = Path("qr_code.png")
        generate_qr(self.page_url, self.qr_code_path)
        LOG.info("Open the following URL on your phone:")
        LOG.info("  %s", self.page_url)
        LOG.info("QR code saved to %s", self.qr_code_path)

    def stop(self):
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None

        if self._ws_loop is not None and self._ws_loop.is_running() and self._ws_stop_event is not None:
            self._ws_loop.call_soon_threadsafe(self._ws_stop_event.set)
            self._ws_loop = None
            self._ws_stop_event = None

    @staticmethod
    def _decode_jpeg(payload: bytes):
        nparr = np.frombuffer(payload, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            LOG.warning("Received invalid JPEG frame")
        return frame


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    receiver = FrameReceiver()
    server = WebSocketFrameServer(receiver)
    server.start()
    print("WebSocket streaming server started.")
    print("Navigate to the generated QR code URL from your phone.")
    print("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down.")


if __name__ == '__main__':
    main()
