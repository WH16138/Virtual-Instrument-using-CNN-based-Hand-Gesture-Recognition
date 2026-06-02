# VisionQuest Development Handover

## Project Summary

VisionQuest is a computer vision-based AR dungeon battle simulator.

The project combines:

1. Markerless AR
2. CNN-based Hand Gesture Recognition
3. Turn-Based Battle System

The application will be implemented primarily in Python using OpenCV.

The focus of the project is computer vision and interaction design rather than game complexity.

---

# Core Requirements

## AR Battlefield

The system should:

1. Detect and register a planar surface.
2. Track the surface using feature matching.
3. Maintain a virtual battlefield aligned with the detected plane.

Preferred approach:

- ORB Feature Detection
- BFMatcher
- RANSAC
- Homography

No ArUco markers or chessboards should be required during gameplay.

---

## Gesture Recognition

Minimum gesture classes:

| Gesture | Action |
|----------|----------|
| Fist | Attack |
| Open Palm | Defend |
| V Sign | Skill |

Requirements:

- Custom CNN model
- Real-time inference
- Stable predictions

---

## Battle System

Turn-based combat.

States:

- Waiting
- Player Turn
- Enemy Turn
- Victory
- Defeat

Player actions:

- Attack
- Defend
- Skill

Enemy actions:

- Simple AI-controlled attack selection

---

## Game Flow

Application Start

↓

User places both hands on a planar surface

↓

Plane registration

↓

AR battlefield generation

↓

Player Turn

↓

Gesture Recognition

↓

Battle Action

↓

Enemy Turn

↓

Victory or Defeat

---

# Development Principles

## Important

This is NOT a game development project.

This is a computer vision project that uses a game as a demonstration platform.

Computer vision features should always take priority over game complexity.

---

# Directory Structure

project/

├── main.py

├── ar/
│   ├── plane_tracker.py
│   ├── homography.py
│   └── ar_renderer.py

├── vision/
│   ├── gesture_detector.py
│   ├── hand_tracker.py
│   └── dataset_collector.py

├── models/
│   ├── gesture_model.keras
│   └── train.py

├── game/
│   ├── game_manager.py
│   ├── battle_system.py
│   ├── player.py
│   ├── enemy.py
│   └── skills.py

├── ui/
│   ├── hud.py
│   ├── damage_text.py
│   └── menu.py

├── assets/

└── README.md

---

# Coding Guidelines

## General

- Keep modules independent.
- Avoid circular imports.
- Separate game logic from rendering logic.
- Separate computer vision logic from gameplay logic.

---

## Vision Layer

Responsible for:

- Camera processing
- Feature extraction
- Gesture recognition
- Hand tracking

Must not contain game logic.

---

## Game Layer

Responsible for:

- Battle calculations
- Turn management
- Player state
- Enemy state

Must not contain OpenCV processing.

---

## Rendering Layer

Responsible for:

- AR overlays
- HUD rendering
- Visual feedback

Must not contain battle calculations.

---

# MVP Requirements

The project is considered complete when:

- Plane registration works
- Markerless AR tracking works
- Three gestures are recognized
- Turn-based combat works
- HP is displayed
- Victory/Defeat states exist

Anything beyond this is optional.

---

# Future Extensions

Potential stretch goals:

- Multiple enemies
- Boss battles
- Gesture-motion combo skills
- Particle effects
- Sound effects
- Inventory system
- Hand-based target selection

These features should only be implemented after the MVP is fully functional.