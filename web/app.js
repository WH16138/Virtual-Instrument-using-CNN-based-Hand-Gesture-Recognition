const video = document.getElementById('video');
const previewImage = document.getElementById('preview');
const stage = document.getElementById('stage');
const statusLabel = document.getElementById('status');
const connectionLabel = document.getElementById('connection');
const framesLabel = document.getElementById('frames');
const previewFramesLabel = document.getElementById('preview-frames');
const qualityLabel = document.getElementById('quality');
const arViewButton = document.getElementById('ar-view');
const cameraViewButton = document.getElementById('camera-view');

const DEFAULT_WS_PORT = 8765;
const JPEG_QUALITY = 0.76;
const SEND_INTERVAL_MS = 34;
const RECONNECT_BASE_DELAY_MS = 1000;
const RECONNECT_MAX_DELAY_MS = 5000;
const MAX_BUFFERED_BYTES = 450000;
const MAX_FRAME_WIDTH = 960;

const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

let ws = null;
let frameCount = 0;
let previewFrameCount = 0;
let isReady = false;
let encodeInFlight = false;
let lastSendTime = 0;
let reconnectTimer = null;
let reconnectAttempts = 0;
let currentPreviewUrl = null;

function getWebSocketPort() {
    const params = new URLSearchParams(window.location.search);
    const configuredPort = Number.parseInt(params.get('ws_port'), 10);
    return Number.isInteger(configuredPort) && configuredPort > 0 ? configuredPort : DEFAULT_WS_PORT;
}

function updateStatus(text) {
    statusLabel.textContent = text;
}

function updateConnection(connected) {
    connectionLabel.textContent = connected ? 'Connected' : 'Disconnected';
    connectionLabel.classList.toggle('connected', connected);
}

function updateFrames() {
    framesLabel.textContent = `Sent ${frameCount}`;
    previewFramesLabel.textContent = `Preview ${previewFrameCount}`;
}

function setViewMode(showCamera) {
    stage.classList.toggle('camera-mode', showCamera);
    arViewButton.classList.toggle('active', !showCamera);
    cameraViewButton.classList.toggle('active', showCamera);
}

function showRenderedPreview(data) {
    const blob = data instanceof Blob ? data : new Blob([data], { type: 'image/jpeg' });
    if (currentPreviewUrl) {
        URL.revokeObjectURL(currentPreviewUrl);
    }
    currentPreviewUrl = URL.createObjectURL(blob);
    previewImage.src = currentPreviewUrl;
    previewFrameCount += 1;
    updateFrames();
    updateStatus('PC-rendered AR preview active.');
}

function scheduleReconnect() {
    if (reconnectTimer !== null) {
        return;
    }

    const delay = Math.min(RECONNECT_BASE_DELAY_MS * (reconnectAttempts + 1), RECONNECT_MAX_DELAY_MS);
    reconnectAttempts += 1;
    updateStatus(`Reconnecting in ${Math.round(delay / 1000)}s...`);

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
    ws.binaryType = 'blob';

    ws.onopen = () => {
        reconnectAttempts = 0;
        updateConnection(true);
        updateStatus('Camera connected. Waiting for PC preview...');
    };

    ws.onmessage = (event) => {
        showRenderedPreview(event.data);
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
            video: {
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30, max: 30 }
            },
            audio: false
        });
        video.srcObject = stream;
        await video.play();
        isReady = true;
        updateStatus('Camera ready. Connecting...');
        initWebSocket();
        requestAnimationFrame(captureLoop);
    } catch (err) {
        console.error(err);
        updateStatus('Camera permission is required. Check browser settings and reload.');
    }
}

function captureLoop(timestamp) {
    if (
        isReady
        && ws
        && ws.readyState === WebSocket.OPEN
        && !encodeInFlight
        && timestamp - lastSendTime >= SEND_INTERVAL_MS
        && ws.bufferedAmount < MAX_BUFFERED_BYTES
    ) {
        lastSendTime = timestamp;
        sendCurrentFrame();
    }

    requestAnimationFrame(captureLoop);
}

function sendCurrentFrame() {
    if (video.videoWidth === 0 || video.videoHeight === 0) {
        return;
    }

    encodeInFlight = true;
    const scale = Math.min(1, MAX_FRAME_WIDTH / video.videoWidth);
    const targetWidth = Math.round(video.videoWidth * scale);
    const targetHeight = Math.round(video.videoHeight * scale);
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
    }
    ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
    canvas.toBlob((blob) => {
        if (blob && ws && ws.readyState === WebSocket.OPEN) {
            ws.send(blob);
            frameCount += 1;
            updateFrames();
        }
        encodeInFlight = false;
    }, 'image/jpeg', JPEG_QUALITY);
}

arViewButton.addEventListener('click', () => setViewMode(false));
cameraViewButton.addEventListener('click', () => setViewMode(true));
window.addEventListener('beforeunload', () => {
    if (currentPreviewUrl) {
        URL.revokeObjectURL(currentPreviewUrl);
    }
});

qualityLabel.textContent = `JPEG ${JPEG_QUALITY} | max ${MAX_FRAME_WIDTH}px`;
updateConnection(false);
updateFrames();
setViewMode(false);
initCamera();