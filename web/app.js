const video = document.getElementById('video');
const statusLabel = document.getElementById('status');
const connectionLabel = document.getElementById('connection');
const framesLabel = document.getElementById('frames');
const qualityLabel = document.getElementById('quality');

const WS_PORT = 8765;
const JPEG_QUALITY = 0.7;
const SEND_INTERVAL_MS = 100;
let ws = null;
let frameCount = 0;
let isReady = false;
let lastSendTime = 0;

function updateStatus(text) {
    statusLabel.textContent = text;
}

function updateConnection(connected) {
    connectionLabel.textContent = connected ? 'WebSocket: 연결됨' : 'WebSocket: 연결되지 않음';
    connectionLabel.style.background = connected ? '#154' : '#311';
}

function updateFrames(count) {
    framesLabel.textContent = `전송 프레임: ${count}`;
}

function initWebSocket() {
    const host = window.location.hostname;
    const wsUrl = `ws://${host}:${WS_PORT}`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        updateConnection(true);
        updateStatus('카메라가 연결되었습니다. 프레임 전송 중...');
    };

    ws.onclose = () => {
        updateConnection(false);
        updateStatus('WebSocket 연결이 끊겼습니다. 페이지를 새로고침하세요.');
    };

    ws.onerror = () => {
        updateConnection(false);
        updateStatus('WebSocket 오류가 발생했습니다. 서버를 확인하세요.');
    };
}

async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
        video.srcObject = stream;
        await video.play();
        isReady = true;
        updateStatus('카메라가 활성화되었습니다. 프레임을 전송합니다.');
        initWebSocket();
        requestAnimationFrame(captureLoop);
    } catch (err) {
        console.error(err);
        updateStatus('카메라 권한이 필요합니다. 브라우저 설정을 확인하세요.');
    }
}

function captureLoop(timestamp) {
    if (!isReady || !ws || ws.readyState !== WebSocket.OPEN) {
        requestAnimationFrame(captureLoop);
        return;
    }

    if (timestamp - lastSendTime >= SEND_INTERVAL_MS && ws.bufferedAmount < 500000) {
        lastSendTime = timestamp;
        sendCurrentFrame();
    }

    requestAnimationFrame(captureLoop);
}

function sendCurrentFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
        if (blob && ws && ws.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then((buffer) => {
                ws.send(buffer);
                frameCount += 1;
                updateFrames(frameCount);
            });
        }
    }, 'image/jpeg', JPEG_QUALITY);
}

qualityLabel.textContent = `화질: ${JPEG_QUALITY}`;
initCamera();
