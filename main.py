"""
========================================================
  Real-Time Sign Language Detection System
  FILE: main.py
  PURPOSE: Application entry point.
           Stitches together:
             • SignPredictor   (predict.py)
             • VoiceEngine     (app/voice_engine.py)
             • UIRenderer      (app/ui.py)
           and runs the real-time webcam loop.

  HOW TO RUN:
      python main.py

  KEYBOARD SHORTCUTS:
      Q  — Quit
      V  — Toggle voice on / off
      S  — Take a screenshot and save to screenshots/
      H  — Show / hide help overlay
========================================================
"""

import cv2
import os
import sys
import time

# Local modules
from predict           import SignPredictor
from app.voice_engine  import VoiceEngine
from app.ui            import UIRenderer


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
WEBCAM_INDEX         = 0        # Change if your camera isn't /dev/video0
FRAME_WIDTH          = 640
FRAME_HEIGHT         = 480
CONFIDENCE_THRESHOLD = 65.0     # % — below this threshold, ignore prediction
SCREENSHOT_DIR       = "screenshots"
WINDOW_TITLE         = "Sign Language Detection System"


# ─────────────────────────────────────────────
#  HELP OVERLAY TEXT
# ─────────────────────────────────────────────
HELP_LINES = [
    "  KEYBOARD SHORTCUTS",
    "  ─────────────────────",
    "  Q  :  Quit",
    "  V  :  Toggle voice on / off",
    "  S  :  Save screenshot",
    "  H  :  Toggle this help panel",
]


# ─────────────────────────────────────────────
#  HELPER: overlay semi-transparent help panel
# ─────────────────────────────────────────────
def draw_help(frame):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 50), (300, 200), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    for i, line in enumerate(HELP_LINES):
        y = 75 + i * 22
        cv2.putText(frame, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (200, 220, 255), 1, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────
#  MAIN APPLICATION LOOP
# ─────────────────────────────────────────────
def run():
    print("=" * 55)
    print("  Real-Time Sign Language Detection System")
    print("=" * 55)

    # ── 1. Load model ──────────────────────────────────────────
    print("\n[INFO] Loading model…")
    try:
        predictor = SignPredictor()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n")
        print("Steps to fix:")
        print("  1.  python collect_data.py   (gather training images)")
        print("  2.  python train_model.py    (train the CNN)")
        print("  3.  python main.py           (run the app)\n")
        sys.exit(1)

    # ── 2. Voice engine ────────────────────────────────────────
    print("[INFO] Initialising voice engine…")
    voice   = VoiceEngine(hold_seconds=1.5, cooldown_seconds=3.0)
    voice_on = True
    if not voice.available:
        print("[WARN] pyttsx3 unavailable — voice output disabled.")

    # ── 3. UI renderer ─────────────────────────────────────────
    ui = UIRenderer(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)

    # ── 4. Open webcam ──────────────────────────────────────────
    print(f"[INFO] Opening webcam (index {WEBCAM_INDEX})…")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam.")
        print("  • Check if another app is using the camera.")
        print("  • Try changing WEBCAM_INDEX in main.py.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ── 5. Screenshot folder ────────────────────────────────────
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    # ── 6. State variables ──────────────────────────────────────
    show_help     = False
    last_label    = ""
    last_conf     = 0.0
    last_bbox     = None

    print("\n[INFO] System running. Press Q to quit, H for help.\n")
    voice.speak_now("Sign Language Detection System started.")

    # ─────────────────────────────────────────
    #  MAIN LOOP
    # ─────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        # Mirror frame (more intuitive for the user)
        frame = cv2.flip(frame, 1)

        # ── Prediction ──
        label, conf, annotated_frame, bbox = predictor.predict_frame(frame)

        # Apply confidence threshold — ignore shaky low-confidence guesses
        if conf < CONFIDENCE_THRESHOLD:
            label = ""
            conf  = 0.0
            bbox  = None

        # Cache last valid prediction (avoid flickering when hand briefly lost)
        if label:
            last_label = label
            last_conf  = conf
            last_bbox  = bbox
        else:
            # Fade out after 1 second of no detection
            last_label = ""

        # ── Voice update ──
        if voice_on and voice.available and last_label:
            voice.update(last_label, last_conf,
                         confidence_threshold=CONFIDENCE_THRESHOLD)
        else:
            voice.status = "Muted" if not voice_on else "No sign"

        # ── Draw UI ──
        output = ui.render(
            annotated_frame,
            label       = last_label,
            confidence  = last_conf,
            voice_status= voice.status,
            bbox        = last_bbox,
        )

        # Muted indicator
        if not voice_on:
            cv2.putText(output, "[VOICE OFF]", (FRAME_WIDTH - 140, FRAME_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1, cv2.LINE_AA)

        # Help overlay
        if show_help:
            output = draw_help(output)

        cv2.imshow(WINDOW_TITLE, output)

        # ── Key Handling ──
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:          # Q / ESC → Quit
            print("[INFO] Quit signal received.")
            break

        elif key == ord('v'):                      # V → Toggle voice
            voice_on = not voice_on
            state = "ON" if voice_on else "OFF"
            print(f"[INFO] Voice {state}")
            voice.speak_now(f"Voice turned {state}") if voice_on else None

        elif key == ord('s'):                      # S → Screenshot
            ts       = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SCREENSHOT_DIR, f"sign_{ts}.png")
            cv2.imwrite(filename, output)
            print(f"[INFO] Screenshot saved → {filename}")

        elif key == ord('h'):                      # H → Toggle help
            show_help = not show_help

    # ── Cleanup ─────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    predictor.close()
    voice.shutdown()
    print("[INFO] Application closed cleanly.")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run()
