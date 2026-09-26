# HandTrack

A real-time, gesture-controlled canvas system built with OpenCV and MediaPipe. Use hand gestures to draw text boxes on a live webcam feed, type into them with your keyboard, drag them around, and delete them — no mouse required. A face-tracking layer also detects your mood (happy/sad/neutral) from your expression and drives a small canned-response "AI" chat feature.

## How it works

- **Pinch with both hands** to draw a new canvas (rectangle) on screen.
- **Hover one index finger** over a canvas to select it for typing — then just type on your keyboard and the text appears inside the box, wrapped to fit.
- **Peace sign** (index + middle finger extended) over a canvas lets you drag it around.
- **Three fingers** (index + middle + ring) hovering over a canvas deletes it.
- **Open hand** hovering over your selected canvas deselects it — and if it was a "You" canvas with unanswered text, triggers a canned reply canvas from the "AI", tagged and color-coded differently, using simple keyword matching (greeting/thanks/question/goodbye/default) filtered through your detected mood.
- **Two index fingers held up at once** toggles a settings panel (visible only while a canvas is selected) for changing that canvas's text color, background color, and "speaker" tag (You / AI).
- A face-landmarker model tracks your expression in the corner of the frame and classifies your mood from smile/frown blendshape scores, which then flavors which canned replies get picked.

## Why I built it

I wanted to explore gesture-only interaction as an input method — turning a webcam into a lightweight, mouse-free "canvas" interface, and layering in some social/emotional context (mood detection) to make the interaction feel more responsive and alive rather than purely mechanical.

## Tech stack

- Python
- [OpenCV](https://opencv.org/) for video capture, rendering, and the UI overlay
- [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/vision) — `HandLandmarker` (up to 2 hands) and `FaceLandmarker` with blendshapes (mood detection)

Gestures are stabilized frame-to-frame with a custom tracker that debounces flicker (a gesture must hold for a few consecutive frames to register), tolerates brief occlusion, and smooths point positions so nothing jitters.

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/omarbenayed05-maker/HandTrack.git
   cd HandTrack
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe
   ```
3. Make sure `hand_landmarker.task` and `face_landmarker.task` are present in the project folder (already included in this repo). If you ever need to re-download the face model, get it from:
   `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`
4. Run it:
   ```bash
   python HandTrack.py
   ```
5. Press `Esc` to quit at any time (or `q` when nothing is selected).

## Roadmap

- [ ] Replace keyboard typing with speech-to-text input (original concept)
- [ ] Persist canvases between sessions
- [ ] Expand the "AI" response system beyond canned keyword-based replies

