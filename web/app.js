const video = document.getElementById('video');
const statusLabel = document.getElementById('status');
const connectionLabel = document.getElementById('connection');
const framesLabel = document.getElementById('frames');
const qualityLabel = document.getElementById('quality');

const DEFAULT_WS_PORT = 8765;
const JPEG_QUALITY = 0.7;
const SEND_INTERVAL_MS = 100;
const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 5000;
const MAX_BUFFERED_BYTES = 500000;
const MAX_FRAME_WIDTH = 640;

const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

let ws = null;
let frameCount = 0;
let isReady = false;
let lastSendTime = 0;
let reconnectTimer = null;
let reconnectAttempts = 0;

function getWebSocketPort() {
    const params = new URLSearchParams(window.location.search);
    const configuredPort = Number.parseInt(params.get('ws_port'), 10);
    return Number.isInteger(configuredPort) && configuredPort > 0 ? configuredPort : DEFAULT_WS_PORT;
}

function updateStatus(text) {
    statusLabel.textContent = text;
}

function updateConnection(connected) {
    connectionLabel.textContent = connected ? 'WebSocket: connected' : 'WebSocket: disconnected';
    connectionLabel.style.background = connected ? '#154' : '#311';
}

function updateFrames(count) {
    framesLabel.textContent = `Frames sent: ${count}`;
}

function scheduleReconnect() {
    if (reconnectTimer !== null) {
        return;
    }

    const delay = Math.min(RECONNECT_BASE_DELAY_MS * (reconnectAttempts + 1), RECONNECT_MAX_DELAY_MS);
    reconnectAttempts += 1;
    updateStatus(`WebSocket disconnected. Reconnecting in ${Math.round(delay / 1000)}s...`);

    reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;
        initWebSocket();
    }, delay);
}

function initWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return;
    }

    const host = window.location.hostname;
    const wsUrl = `ws://${host}:${getWebSocketPort()}`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        reconnectAttempts = 0;
        updateConnection(true);
        updateStatus('Camera connected. Streaming frames...');
    };

    ws.onclose = () => {
        updateConnection(false);
        scheduleReconnect();
    };

    ws.onerror = () => {
        updateConnection(false);
        if (ws && ws.readyState !== WebSocket.CLOSED) {
            ws.close();
        }
    };
}

async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
            audio: false
        });
        video.srcObject = stream;
        await video.play();
        isReady = true;
        updateStatus('Camera is ready. Connecting to WebSocket...');
        initWebSocket();
        requestAnimationFrame(captureLoop);
    } catch (err) {
        console.error(err);
        updateStatus('Camera permission is required. Check browser settings and reload this page.');
    }
}

function captureLoop(timestamp) {
    if (!isReady || !ws || ws.readyState !== WebSocket.OPEN) {
        requestAnimationFrame(captureLoop);
        return;
    }

    if (timestamp - lastSendTime >= SEND_INTERVAL_MS && ws.bufferedAmount < MAX_BUFFERED_BYTES) {
        lastSendTime = timestamp;
        sendCurrentFrame();
    }

    requestAnimationFrame(captureLoop);
}

function sendCurrentFrame() {
    if (video.videoWidth === 0 || video.videoHeight === 0) {
        return;
    }

    const scale = Math.min(1, MAX_FRAME_WIDTH / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
        if (blob && ws && ws.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then((buffer) => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(buffer);
                    frameCount += 1;
                    updateFrames(frameCount);
                }
            });
        }
    }, 'image/jpeg', JPEG_QUALITY);
}

qualityLabel.textContent = `JPEG quality: ${JPEG_QUALITY}`;
updateConnection(false);
updateFrames(frameCount);
initCamera();
