import sys
import types

# --- Work around Windows blocking matplotlib's ft2font DLL (Smart App Control /
# Application Control policy). mediapipe imports matplotlib.pyplot internally
# for legacy drawing helpers this script never calls, so we hand it a harmless
# stand-in module before it gets the chance to load the real thing. ---
for _name in ("matplotlib", "matplotlib.pyplot", "matplotlib.cm", "matplotlib.colors"):
    if _name not in sys.modules:
        sys.modules[_name] = types.ModuleType(_name)
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
sys.modules["matplotlib"].cm = sys.modules["matplotlib.cm"]
sys.modules["matplotlib"].colors = sys.modules["matplotlib.colors"]

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import random

# --- MediaPipe setup ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

# Face landmarker (with blendshapes) for mood detection — download from:
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# and place it next to this script as face_landmarker.task
face_base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    num_faces=1,
    output_face_blendshapes=True,
    min_face_detection_confidence=0.25,
    min_face_presence_confidence=0.25,
    min_tracking_confidence=0.25,
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

cap = cv2.VideoCapture(0)

PINCH_THRESHOLD = 40    # pixels — tune if pinch feels too sensitive/insensitive
MIN_BOX_SIZE = 30       # pixels — ignore tiny/accidental boxes from flicker

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
LINE_PADDING = 6        # vertical gap between wrapped lines
TEXT_MARGIN = 10        # inner margin from the canvas border

# --- Gesture stabilization tuning ---
CONFIRM_FRAMES = 3      # a gesture must be seen this many consecutive matched frames before it "counts"
MISS_TOLERANCE = 5      # frames a gesture can briefly vanish (occlusion/misdetection) and still be held
SMOOTH_ALPHA = 0.35     # 0-1, lower = smoother/laggier, higher = snappier/jitterier
MATCH_DIST = 80         # px — max jump between frames to still be considered the same point

# --- Color palettes for the config panel (BGR tuples; None = transparent) ---
TEXT_COLOR_OPTIONS = [
    ("White", (255, 255, 255)),
    ("Black", (0, 0, 0)),
    ("Red", (0, 0, 255)),
    ("Green", (0, 255, 0)),
    ("Blue", (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
    ("Cyan", (255, 255, 0)),
    ("Magenta", (255, 0, 255)),
]

BG_COLOR_OPTIONS = [
    ("None", None),
    ("Black", (0, 0, 0)),
    ("White", (255, 255, 255)),
    ("Red", (0, 0, 255)),
    ("Green", (0, 255, 0)),
    ("Blue", (255, 0, 0)),
    ("Gray", (80, 80, 80)),
    ("Navy", (90, 40, 0)),
]

SWATCH_SIZE = 46
SWATCH_GAP = 14

# --- Speaker / role options for the panel ---
SPEAKER_OPTIONS = [
    ("You", 'user'),
    ("AI", 'ai'),
]

USER_BORDER_COLOR = (255, 200, 0)   # blue-ish — "You" canvases
AI_BORDER_COLOR = (200, 0, 255)     # purple-ish — "AI" canvases

# --- Mood detection tuning ---
MOOD_SMOOTH_ALPHA = 0.15    # smooths the smile/frown score over time
MOOD_HAPPY_THRESHOLD = 0.15
MOOD_SAD_THRESHOLD = -0.05

# --- Playful canned "AI" responses, keyed by category + current mood ---
RESPONSE_BANK = {
    'greeting': {
        'happy': ["Hey hey! Love that smile!", "Hi there, you're glowing today!", "Hello! Great energy!"],
        'sad': ["Hi... I'm here if you need to talk.", "Hey, hope things look up soon.", "Hello, take it easy today."],
        'neutral': ["Hey there!", "Hello!", "Hi, what's up?"],
    },
    'thanks': {
        'happy': ["Anytime! You're on fire today!", "You're so welcome!!", "Glad I could help, happy human!"],
        'sad': ["Of course. Sending a little extra support your way.", "You're welcome — hang in there.", "Anytime, be gentle with yourself."],
        'neutral': ["You're welcome.", "No problem at all.", "Happy to help."],
    },
    'question': {
        'happy': ["Ooh good question, let's dig in!", "I love a good puzzle!", "Great question, ask away!"],
        'sad': ["That's a fair question, let's take it slow.", "Good question, no pressure to have it figured out.", "Let's think it through gently."],
        'neutral': ["Interesting question.", "Let me think about that.", "Good question."],
    },
    'bye': {
        'happy': ["Bye! Keep that energy up!", "See ya! You're crushing it!", "Later, stay awesome!"],
        'sad': ["Take care of yourself, okay?", "Bye for now, be kind to yourself.", "See you soon, hang in there."],
        'neutral': ["Goodbye.", "See you later.", "Bye!"],
    },
    'default': {
        'happy': ["Love that energy!", "Nice, tell me more!", "That's awesome!"],
        'sad': ["I hear you, that sounds tough.", "Thanks for sharing that with me.", "I'm listening."],
        'neutral': ["Got it.", "Noted!", "I see."],
    },
}


def classify_message(text):
    t = text.strip().lower()
    if not t:
        return 'default'
    if any(t.startswith(w) for w in ('hi', 'hello', 'hey', 'yo', 'sup')):
        return 'greeting'
    if 'thank' in t or 'thx' in t:
        return 'thanks'
    if any(w in t for w in ('bye', 'goodbye', 'see ya', 'later')):
        return 'bye'
    if t.endswith('?'):
        return 'question'
    return 'default'


def generate_reply(text, mood):
    category = classify_message(text)
    options_list = RESPONSE_BANK[category][mood]
    return random.choice(options_list)


# --- Gesture tracker: smooths + debounces a set of same-type points frame to frame ---
class GestureTracker:
    """
    Takes each frame's raw detected points for one gesture type (e.g. all pinch
    points across both hands) and turns them into stable, smoothed points:
      - a point must be seen for CONFIRM_FRAMES consecutive matched frames
        before it's reported (kills single-frame misclassification flicker)
      - a point that briefly stops being detected is still reported (frozen at
        its last position) for up to MISS_TOLERANCE frames (kills dropout flicker)
      - reported positions are exponentially smoothed (kills jitter in the dot)
    """

    def __init__(self):
        self.tracks = []  # each: {'pos': (x, y), 'seen': int, 'missed': int}

    def update(self, detections):
        used = set()

        for track in self.tracks:
            best_i, best_d = None, None
            for i, d in enumerate(detections):
                if i in used:
                    continue
                dist = math.hypot(d[0] - track['pos'][0], d[1] - track['pos'][1])
                if dist <= MATCH_DIST and (best_d is None or dist < best_d):
                    best_d, best_i = dist, i

            if best_i is not None:
                used.add(best_i)
                nx, ny = detections[best_i]
                px, py = track['pos']
                track['pos'] = (px * (1 - SMOOTH_ALPHA) + nx * SMOOTH_ALPHA,
                                 py * (1 - SMOOTH_ALPHA) + ny * SMOOTH_ALPHA)
                track['seen'] += 1
                track['missed'] = 0
            else:
                track['missed'] += 1

        self.tracks = [t for t in self.tracks if t['missed'] <= MISS_TOLERANCE]

        for i, d in enumerate(detections):
            if i not in used:
                self.tracks.append({'pos': (float(d[0]), float(d[1])), 'seen': 1, 'missed': 0})

        return [(int(t['pos'][0]), int(t['pos'][1])) for t in self.tracks if t['seen'] >= CONFIRM_FRAMES]


pinch_tracker = GestureTracker()
peace_tracker = GestureTracker()
index_tracker = GestureTracker()
three_tracker = GestureTracker()
open_tracker = GestureTracker()

# --- Persistent state ---
canvases = []            # list of {'rect', 'text', 'text_color', 'bg_color'}
in_progress_rect = None  # rectangle being actively drawn (both hands pinching)
was_pinching_both = False

dragging_index = None    # index into canvases currently being dragged
prev_drag_point = None

selected_canvas_index = None  # canvas currently selected for keyboard typing

panel_open = False       # is the color-config panel showing?
panel_target = None      # which canvas index the panel is editing
was_two_index = False    # edge-detect for the two-index-fingers toggle gesture
panel_pinch_active = False  # edge-detect for single-hand pinch inside the panel

mood_score = 0.0         # smoothed smile-minus-frown score
current_mood = 'neutral'  # 'happy' / 'sad' / 'neutral'


# --- Gesture helper functions (return raw, per-hand points; no smoothing here) ---
def get_pinch_point(hand_landmarks, w, h):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
    x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
    distance = math.hypot(x2 - x1, y2 - y1)
    if distance < PINCH_THRESHOLD:
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    return None


def _tip_above_pip(hand_landmarks, tip_idx, pip_idx, h):
    tip_y = hand_landmarks[tip_idx].y * h
    pip_y = hand_landmarks[pip_idx].y * h
    return tip_y < pip_y  # extended = tip higher (smaller y) than pip joint


def is_peace_sign(hand_landmarks, w, h):
    """Index + middle extended, ring + pinky curled."""
    index_e = _tip_above_pip(hand_landmarks, 8, 6, h)
    middle_e = _tip_above_pip(hand_landmarks, 12, 10, h)
    ring_c = not _tip_above_pip(hand_landmarks, 16, 14, h)
    pinky_c = not _tip_above_pip(hand_landmarks, 20, 18, h)
    if index_e and middle_e and ring_c and pinky_c:
        x1, y1 = hand_landmarks[8].x * w, hand_landmarks[8].y * h
        x2, y2 = hand_landmarks[12].x * w, hand_landmarks[12].y * h
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    return None


def is_index_only(hand_landmarks, w, h):
    """Only index extended, middle/ring/pinky curled — select / config-toggle trigger."""
    index_e = _tip_above_pip(hand_landmarks, 8, 6, h)
    middle_c = not _tip_above_pip(hand_landmarks, 12, 10, h)
    ring_c = not _tip_above_pip(hand_landmarks, 16, 14, h)
    pinky_c = not _tip_above_pip(hand_landmarks, 20, 18, h)
    if index_e and middle_c and ring_c and pinky_c:
        return (int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h))
    return None


def is_three_fingers(hand_landmarks, w, h):
    """Index + middle + ring extended, pinky curled — hover-to-delete trigger."""
    index_e = _tip_above_pip(hand_landmarks, 8, 6, h)
    middle_e = _tip_above_pip(hand_landmarks, 12, 10, h)
    ring_e = _tip_above_pip(hand_landmarks, 16, 14, h)
    pinky_c = not _tip_above_pip(hand_landmarks, 20, 18, h)
    if index_e and middle_e and ring_e and pinky_c:
        x = int((hand_landmarks[8].x + hand_landmarks[12].x + hand_landmarks[16].x) / 3 * w)
        y = int((hand_landmarks[8].y + hand_landmarks[12].y + hand_landmarks[16].y) / 3 * h)
        return (x, y)
    return None


def is_open_hand(hand_landmarks, w, h):
    """All four fingers extended (thumb ignored) — deselect/cancel trigger."""
    index_e = _tip_above_pip(hand_landmarks, 8, 6, h)
    middle_e = _tip_above_pip(hand_landmarks, 12, 10, h)
    ring_e = _tip_above_pip(hand_landmarks, 16, 14, h)
    pinky_e = _tip_above_pip(hand_landmarks, 20, 18, h)
    if index_e and middle_e and ring_e and pinky_e:
        xs = [hand_landmarks[i].x for i in (0, 8, 12, 16, 20)]
        ys = [hand_landmarks[i].y for i in (0, 8, 12, 16, 20)]
        return (int(sum(xs) / len(xs) * w), int(sum(ys) / len(ys) * h))
    return None


def point_in_rect(point, rect):
    x, y = point
    x1, y1, x2, y2 = rect
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    return left <= x <= right and top <= y <= bottom


def get_mood_score(face_result):
    """Smile score minus frown score from face blendshapes, or None if no face."""
    if not face_result.face_blendshapes:
        return None
    categories = {c.category_name: c.score for c in face_result.face_blendshapes[0]}
    smile = (categories.get('mouthSmileLeft', 0) + categories.get('mouthSmileRight', 0)) / 2
    frown = (categories.get('mouthFrownLeft', 0) + categories.get('mouthFrownRight', 0)) / 2
    return smile - frown


def draw_mood_icon(frame, mood, center, radius=30):
    color = (0, 255, 255) if mood == 'happy' else ((255, 100, 100) if mood == 'sad' else (200, 200, 200))
    cx, cy = center
    cv2.circle(frame, (cx, cy), radius, color, 2)
    eye_dx = radius // 3
    eye_y = cy - radius // 4
    cv2.circle(frame, (cx - eye_dx, eye_y), 3, color, -1)
    cv2.circle(frame, (cx + eye_dx, eye_y), 3, color, -1)
    mouth_y = cy + radius // 4
    if mood == 'happy':
        cv2.ellipse(frame, (cx, mouth_y - 5), (radius // 2, radius // 3), 0, 0, 180, color, 2)
    elif mood == 'sad':
        cv2.ellipse(frame, (cx, mouth_y + 8), (radius // 2, radius // 3), 0, 180, 360, color, 2)
    else:
        cv2.line(frame, (cx - radius // 2, mouth_y), (cx + radius // 2, mouth_y), color, 2)
    cv2.putText(frame, mood.capitalize(), (cx - radius, cy + radius + 20), FONT, 0.5, color, 1, cv2.LINE_AA)


def draw_face_landmarks(frame, face_result, w, h):
    """Show the detected face mesh so face tracking is visible in the preview."""
    if not face_result.face_landmarks:
        return

    for landmark in face_result.face_landmarks[0]:
        x = max(0, min(w - 1, int(landmark.x * w)))
        y = max(0, min(h - 1, int(landmark.y * h)))
        cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)


# --- Text handling: keep everything clipped inside the canvas rect ---
def wrap_text_to_width(text, max_width):
    lines = []
    for paragraph in text.split('\n'):
        if paragraph == '':
            lines.append('')
            continue
        words = paragraph.split(' ')
        current = ''
        for word in words:
            candidate = word if current == '' else current + ' ' + word
            (tw, _), _ = cv2.getTextSize(candidate, FONT, FONT_SCALE, FONT_THICKNESS)
            if tw <= max_width or current == '':
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_canvas_box(frame, canvas, is_selected):
    x1, y1, x2, y2 = canvas['rect']
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)

    if canvas['bg_color'] is not None:
        cv2.rectangle(frame, (left, top), (right, bottom), canvas['bg_color'], -1)

    if is_selected:
        border_color = (0, 255, 255)
    else:
        border_color = AI_BORDER_COLOR if canvas['role'] == 'ai' else USER_BORDER_COLOR
    cv2.rectangle(frame, (left, top), (right, bottom), border_color, 2)

    tag = "AI" if canvas['role'] == 'ai' else "You"
    cv2.putText(frame, tag, (left + 6, top + 16), FONT, 0.5, border_color, 1, cv2.LINE_AA)
    tag_height = 22

    text = canvas['text']
    if not text:
        return
    max_width = max(1, (right - left) - 2 * TEXT_MARGIN)
    max_height = max(1, (bottom - top) - tag_height - 2 * TEXT_MARGIN)

    (_, line_h), _ = cv2.getTextSize("Ag", FONT, FONT_SCALE, FONT_THICKNESS)
    line_step = line_h + LINE_PADDING

    lines = wrap_text_to_width(text, max_width)
    max_lines = max(1, max_height // line_step)
    lines = lines[:max_lines]

    clip_x1 = max(left + TEXT_MARGIN, 0)
    clip_y1 = max(top + tag_height + TEXT_MARGIN, 0)
    clip_x2 = min(right - TEXT_MARGIN, frame.shape[1])
    clip_y2 = min(bottom - TEXT_MARGIN, frame.shape[0])
    if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
        return

    roi = frame[clip_y1:clip_y2, clip_x1:clip_x2]
    for i, line in enumerate(lines):
        text_y = line_h + i * line_step
        text_x = 0
        if 0 <= text_y - line_h <= (clip_y2 - clip_y1):
            cv2.putText(roi, line, (text_x, text_y), FONT, FONT_SCALE,
                        canvas['text_color'], FONT_THICKNESS, cv2.LINE_AA)


# --- Color / role config panel ---
def build_panel_layout(w, h):
    n_swatches = max(len(TEXT_COLOR_OPTIONS), len(BG_COLOR_OPTIONS))
    row_width = n_swatches * SWATCH_SIZE + (n_swatches - 1) * SWATCH_GAP
    panel_w = row_width + 60
    panel_h = 320
    panel_x1 = (w - panel_w) // 2
    panel_y1 = (h - panel_h) // 2
    panel_x2 = panel_x1 + panel_w
    panel_y2 = panel_y1 + panel_h

    text_row_y = panel_y1 + 70
    bg_row_y = panel_y1 + 160
    speaker_row_y = panel_y1 + 250

    swatches = []
    start_x = panel_x1 + 30
    for i, (_, color) in enumerate(TEXT_COLOR_OPTIONS):
        sx1 = start_x + i * (SWATCH_SIZE + SWATCH_GAP)
        sy1 = text_row_y
        swatches.append(((sx1, sy1, sx1 + SWATCH_SIZE, sy1 + SWATCH_SIZE), 'text', color))
    for i, (_, color) in enumerate(BG_COLOR_OPTIONS):
        sx1 = start_x + i * (SWATCH_SIZE + SWATCH_GAP)
        sy1 = bg_row_y
        swatches.append(((sx1, sy1, sx1 + SWATCH_SIZE, sy1 + SWATCH_SIZE), 'bg', color))

    speaker_width = 110
    for i, (_, value) in enumerate(SPEAKER_OPTIONS):
        sx1 = start_x + i * (speaker_width + SWATCH_GAP)
        sy1 = speaker_row_y
        swatches.append(((sx1, sy1, sx1 + speaker_width, sy1 + SWATCH_SIZE), 'speaker', value))

    return (panel_x1, panel_y1, panel_x2, panel_y2), text_row_y, bg_row_y, speaker_row_y, swatches


def draw_panel(frame, panel_rect, text_row_y, bg_row_y, speaker_row_y, swatches, target_canvas):
    px1, py1, px2, py2 = panel_rect
    overlay = frame.copy()
    cv2.rectangle(overlay, (px1, py1), (px2, py2), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (200, 200, 200), 2)

    cv2.putText(frame, "Canvas Settings  (pinch to pick, two-finger hover to close)",
                (px1 + 20, py1 + 30), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Text color", (px1 + 20, text_row_y - 10), FONT, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Background", (px1 + 20, bg_row_y - 10), FONT, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Speaker", (px1 + 20, speaker_row_y - 10), FONT, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    for rect, target, value in swatches:
        sx1, sy1, sx2, sy2 = rect
        if target == 'speaker':
            fill = AI_BORDER_COLOR if value == 'ai' else USER_BORDER_COLOR
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), fill, -1)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 255, 255), 1)
            label = "AI" if value == 'ai' else "You"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
            cv2.putText(frame, label, (sx1 + (sx2 - sx1 - tw) // 2, sy1 + (sy2 - sy1 + th) // 2),
                        FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            if target_canvas['role'] == value:
                cv2.rectangle(frame, (sx1 - 3, sy1 - 3), (sx2 + 3, sy2 + 3), (0, 255, 255), 2)
            continue

        color = value
        if color is None:
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (90, 90, 90), -1)
            cv2.line(frame, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
            cv2.line(frame, (sx1, sy2), (sx2, sy1), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, -1)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 255, 255), 1)

        current = target_canvas['text_color'] if target == 'text' else target_canvas['bg_color']
        if current == color:
            cv2.rectangle(frame, (sx1 - 3, sy1 - 3), (sx2 + 3, sy2 + 3), (0, 255, 255), 2)


# --- Main loop ---
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)
    face_result = face_detector.detect(mp_image)

    raw_mood = get_mood_score(face_result)
    if raw_mood is not None:
        mood_score = mood_score * (1 - MOOD_SMOOTH_ALPHA) + raw_mood * MOOD_SMOOTH_ALPHA
        if mood_score > MOOD_HAPPY_THRESHOLD:
            current_mood = 'happy'
        elif mood_score < MOOD_SAD_THRESHOLD:
            current_mood = 'sad'
        else:
            current_mood = 'neutral'
    draw_face_landmarks(frame, face_result, w, h)
    draw_mood_icon(frame, current_mood, (w - 60, 60))

    # --- Pass 1: collect this frame's RAW gesture points (no drawing yet) ---
    raw_pinch, raw_peace, raw_index, raw_three, raw_open = [], [], [], [], []

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            pinch = get_pinch_point(hand_landmarks, w, h)
            if pinch:
                raw_pinch.append(pinch)

            peace = is_peace_sign(hand_landmarks, w, h)
            if peace:
                raw_peace.append(peace)

            idx = is_index_only(hand_landmarks, w, h)
            if idx:
                raw_index.append(idx)

            tf = is_three_fingers(hand_landmarks, w, h)
            if tf:
                raw_three.append(tf)

            oh = is_open_hand(hand_landmarks, w, h)
            if oh:
                raw_open.append(oh)

    # --- Pass 2: smooth + debounce each gesture type ---
    pinch_points = pinch_tracker.update(raw_pinch)
    peace_points = peace_tracker.update(raw_peace)
    index_points = index_tracker.update(raw_index)
    three_points = three_tracker.update(raw_three)
    open_points = open_tracker.update(raw_open)

    peace_point = peace_points[0] if peace_points else None
    three_finger_point = three_points[0] if three_points else None
    open_hand_point = open_points[0] if open_points else None

    # --- Draw the stabilized dots ---
    for p in pinch_points:
        cv2.circle(frame, p, 8, (0, 255, 0), -1)       # green = pinching
    if peace_point:
        cv2.circle(frame, peace_point, 8, (0, 165, 255), -1)  # orange = drag
    for p in index_points:
        cv2.circle(frame, p, 8, (255, 0, 0), -1)        # blue = select/config target
    if three_finger_point:
        cv2.circle(frame, three_finger_point, 8, (0, 0, 255), -1)  # red = delete target
    if open_hand_point:
        cv2.circle(frame, open_hand_point, 8, (255, 255, 255), -1)  # white = deselect

    # --- Two index fingers at once: toggle the color config panel ---
    two_index_now = len(index_points) == 2
    if two_index_now and not was_two_index:
        if panel_open:
            panel_open = False
            panel_target = None
        elif selected_canvas_index is not None:
            panel_open = True
            panel_target = selected_canvas_index
    was_two_index = two_index_now

    if panel_open and panel_target is not None:
        panel_rect, text_row_y, bg_row_y, speaker_row_y, swatches = build_panel_layout(w, h)

        single_pinch = pinch_points[0] if len(pinch_points) == 1 else None
        pinch_now = single_pinch is not None
        if pinch_now and not panel_pinch_active:
            for rect, target, value in swatches:
                if point_in_rect(single_pinch, rect):
                    if target == 'text':
                        canvases[panel_target]['text_color'] = value if value is not None else (255, 255, 255)
                    elif target == 'bg':
                        canvases[panel_target]['bg_color'] = value
                    else:
                        canvases[panel_target]['role'] = value
                    break
        panel_pinch_active = pinch_now

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        for i, c in enumerate(canvases):
            draw_canvas_box(frame, c, i == selected_canvas_index)
        draw_panel(frame, panel_rect, text_row_y, bg_row_y, speaker_row_y, swatches, canvases[panel_target])

        cv2.imshow("Hand Tracking", frame)
        continue

    # --- Drawing a new canvas (both hands pinching) ---
    if len(pinch_points) == 2:
        in_progress_rect = (pinch_points[0][0], pinch_points[0][1],
                             pinch_points[1][0], pinch_points[1][1])
        was_pinching_both = True
        dragging_index = None
    else:
        if was_pinching_both and in_progress_rect:
            x1, y1, x2, y2 = in_progress_rect
            if abs(x2 - x1) > MIN_BOX_SIZE and abs(y2 - y1) > MIN_BOX_SIZE:
                canvases.append({
                    'rect': in_progress_rect,
                    'text': '',
                    'text_color': (255, 255, 255),
                    'bg_color': None,
                    'role': 'user',
                    'replied': False,
                })
            in_progress_rect = None
        was_pinching_both = False

    # --- Dragging (peace sign) ---
    if peace_point and len(pinch_points) < 2:
        if dragging_index is None:
            for i, c in enumerate(canvases):
                if point_in_rect(peace_point, c['rect']):
                    dragging_index = i
                    prev_drag_point = peace_point
                    break
        else:
            dx = peace_point[0] - prev_drag_point[0]
            dy = peace_point[1] - prev_drag_point[1]
            x1, y1, x2, y2 = canvases[dragging_index]['rect']
            canvases[dragging_index]['rect'] = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            prev_drag_point = peace_point
    else:
        dragging_index = None

    # --- Select a canvas for typing (single index hover) ---
    if len(index_points) == 1:
        for i, c in enumerate(canvases):
            if point_in_rect(index_points[0], c['rect']):
                selected_canvas_index = i
                break

    # --- Deselect (open hand hovering over the selected canvas) ---
    if open_hand_point and selected_canvas_index is not None:
        c = canvases[selected_canvas_index]
        if point_in_rect(open_hand_point, c['rect']):
            if c['role'] == 'user' and c['text'].strip() and not c['replied']:
                reply_text = generate_reply(c['text'], current_mood)
                x1, y1, x2, y2 = c['rect']
                left, right = min(x1, x2), max(x1, x2)
                top, bottom = min(y1, y2), max(y1, y2)
                box_h = bottom - top
                new_rect = (left, bottom + 20, right, bottom + 20 + box_h)
                canvases.append({
                    'rect': new_rect,
                    'text': reply_text,
                    'text_color': (255, 255, 255),
                    'bg_color': None,
                    'role': 'ai',
                    'replied': True,
                })
                c['replied'] = True
            selected_canvas_index = None

    # --- Hover-to-delete (three fingers) ---
    if three_finger_point:
        deleted_selected = False
        new_canvases = []
        for i, c in enumerate(canvases):
            if point_in_rect(three_finger_point, c['rect']):
                if i == selected_canvas_index:
                    deleted_selected = True
                continue
            new_canvases.append(c)
        canvases = new_canvases
        if deleted_selected:
            selected_canvas_index = None

    # --- Keyboard input: typed characters go into the selected canvas ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC always quits
        break
    elif key != 255 and selected_canvas_index is not None:
        c = canvases[selected_canvas_index]
        if key in (8, 127):
            c['text'] = c['text'][:-1]
        elif key in (13, 10):
            c['text'] += '\n'
        elif 32 <= key <= 126:
            c['text'] += chr(key)
        if c['role'] == 'user':
            c['replied'] = False
    elif key == ord('q') and selected_canvas_index is None:
        break

    # --- Draw everything ---
    if in_progress_rect:
        x1, y1, x2, y2 = in_progress_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    for i, c in enumerate(canvases):
        draw_canvas_box(frame, c, i == selected_canvas_index)

    cv2.imshow("Hand Tracking", frame)

cap.release()
cv2.destroyAllWindows()