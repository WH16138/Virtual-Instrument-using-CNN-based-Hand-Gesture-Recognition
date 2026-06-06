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
DEFAULT_PREVIEW_FPS = 15.0
DEFAULT_PREVIEW_MAX_WIDTH = 960
DEFAULT_PREVIEW_JPEG_QUALITY = 72


class WebSocketFrameServer:
    def __init__(
        self,
        receiver: FrameReceiver,
        http_port: int = DEFAULT_HTTP_PORT,
        ws_port: int = DEFAULT_WS_PORT,
        preview_fps: float = DEFAULT_PREVIEW_FPS,
        preview_max_width: int = DEFAULT_PREVIEW_MAX_WIDTH,
        preview_jpeg_quality: int = DEFAULT_PREVIEW_JPEG_QUALITY,
    ):
        self.receiver = receiver
        self.http_port = http_port
        self.ws_port = ws_port
        self.preview_interval = 1.0 / max(float(preview_fps), 1.0)
        self.preview_max_width = max(int(preview_max_width), 160)
        self.preview_jpeg_quality = int(np.clip(preview_jpeg_quality, 30, 95))
        self.web_root = Path(__file__).resolve().parent.parent / "web"
        self._http_server = None
        self._http_thread = None
        self._ws_server = None
        self._ws_thread = None
        self._ws_loop = None
        self._ws_stop_event = None
        self.page_url = None
        self.qr_code_path = None
        self._preview_lock = threading.Lock()
        self._preview_pending_frame = None
        self._preview_payload = None
        self._preview_version = 0
        self._preview_last_submit = 0.0
        self._preview_wakeup = threading.Event()
        self._preview_stop_event = threading.Event()
        self._preview_thread = None

    def start(self):
        try:
            self._start_preview_encoder()
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
        preview_task = asyncio.create_task(self._preview_sender(websocket))

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
            preview_task.cancel()
            try:
                await preview_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self.receiver.set_client_connected(False)
            LOG.info("Mobile client disconnected: %s", client_ip)

    async def _preview_sender(self, websocket):
        last_version = -1
        while True:
            payload, version = self._get_latest_preview(last_version)
            if payload is not None:
                await websocket.send(payload)
                last_version = version
            await asyncio.sleep(self.preview_interval * 0.5)

    def publish_rendered_frame(self, frame):
        """Queue the newest rendered frame for mobile preview without blocking the main loop."""
        if frame is None or self._preview_thread is None or not self.receiver.is_client_connected():
            return False
        now = time.monotonic()
        with self._preview_lock:
            if now - self._preview_last_submit < self.preview_interval:
                return False
            self._preview_last_submit = now
            self._preview_pending_frame = frame.copy()
        self._preview_wakeup.set()
        return True

    def _start_preview_encoder(self):
        self._preview_stop_event.clear()
        self._preview_thread = threading.Thread(target=self._preview_encoder_loop, daemon=True)
        self._preview_thread.start()

    def _preview_encoder_loop(self):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.preview_jpeg_quality]
        while not self._preview_stop_event.is_set():
            self._preview_wakeup.wait(timeout=0.25)
            self._preview_wakeup.clear()
            if self._preview_stop_event.is_set():
                break

            with self._preview_lock:
                frame = self._preview_pending_frame
                self._preview_pending_frame = None
            if frame is None:
                continue

            height, width = frame.shape[:2]
            if width > self.preview_max_width:
                scale = self.preview_max_width / float(width)
                frame = cv2.resize(
                    frame,
                    (self.preview_max_width, max(1, int(round(height * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            success, encoded = cv2.imencode(".jpg", frame, encode_params)
            if not success:
                continue

            with self._preview_lock:
                self._preview_payload = encoded.tobytes()
                self._preview_version += 1

    def _get_latest_preview(self, previous_version):
        with self._preview_lock:
            if self._preview_payload is None or self._preview_version == previous_version:
                return None, previous_version
            return self._preview_payload, self._preview_version

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
        self._preview_stop_event.set()
        self._preview_wakeup.set()
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=1.0)
            self._preview_thread = None

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
    finally:
        server.stop()


if __name__ == '__main__':
    main()
