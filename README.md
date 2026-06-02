# VisionQuest

A computer vision-based AR dungeon battle simulator controlled entirely through hand gestures.

## Overview

VisionQuest is an interactive AR game that combines markerless augmented reality and hand gesture recognition.

The system detects a planar surface (such as a desk), generates a virtual battlefield on top of it, and allows the player to control combat actions using hand gestures instead of traditional input devices.

The project aims to explore how computer vision techniques can be used to create immersive and intuitive game interactions.

## Features

- Markerless AR battlefield generation on a planar surface
- Real-time hand gesture recognition using a CNN model
- Gesture-based combat controls
  - Attack
  - Defend
  - Skill
- Turn-based dungeon battle system
- Real-time AR overlay and interaction

## Planned Technologies

- Python
- OpenCV
- NumPy
- TensorFlow / Keras
- ORB Feature Detection
- Homography Estimation
- Markerless AR Tracking

## Project Goal

To create a computer vision-driven AR gaming experience that demonstrates the integration of:

- Computer Vision
- Augmented Reality
- Deep Learning
- Human-Computer Interaction

while maintaining a complete and playable application.

## Smartphone Camera Streaming

This project now supports using a smartphone camera as the input source via a local WebSocket stream.

### How it works

1. Run the Python application on the PC.
2. The app starts a static web server and WebSocket frame server.
3. A QR code is generated and saved as `qr_code.png`.
4. Scan the QR code from your phone.
5. The mobile browser requests camera permission and streams JPEG frames to the PC.
6. The existing OpenCV vision pipeline processes the latest frame directly.

### Run the streaming server

```bash
pip install -r requirements.txt
python -m network.websocket_server
```

### Run the game

```bash
python main.py
```

### Notes

- The PC serves the mobile client from `web/index.html`.
- Frames are sent as JPEG over WebSocket and only the newest frame is kept.
- The existing pipeline remains unchanged after the frame source switch.
- If phone camera access fails due to insecure local HTTP, use a browser that supports local network camera access or run the site over HTTPS.
