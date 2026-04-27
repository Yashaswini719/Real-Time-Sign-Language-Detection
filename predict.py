"""
========================================================
  Real-Time Sign Language Detection System
  FILE: predict.py
  PURPOSE: Core prediction engine — loads the trained model,
           processes webcam frames with MediaPipe, returns
           (label, confidence) for each frame.
========================================================
"""

import os
import json
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
MODEL_PATH    = os.path.join("model", "sign_language_model.h5")
METADATA_PATH = os.path.join("model", "model_metadata.json")


# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles


class SignPredictor:
    """
    Wraps the trained Keras model + MediaPipe hands pipeline.
    Call `predict_frame(bgr_frame)` to get (label, confidence, annotated_frame).
    """

    def __init__(self):
        self.model      = None
        self.classes    = []
        self.img_size   = (64, 64)   # Must match training config
        self.hands      = None
        self._load_model()
        self._init_mediapipe()

    # ── Private: Load saved model & metadata ──────────────────────────────
    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'.\n"
                "Run train_model.py first to train the model."
            )
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(
                f"Metadata not found at '{METADATA_PATH}'.\n"
                "Ensure train_model.py completed successfully."
            )

        self.model = tf.keras.models.load_model(MODEL_PATH)

        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)
        self.classes  = meta["classes"]
        self.img_size = tuple(meta["img_size"])

        print(f"[INFO] Model loaded  → {MODEL_PATH}")
        print(f"[INFO] Classes ({len(self.classes)}): {self.classes}")

    # ── Private: Initialise MediaPipe Hands ───────────────────────────────
    def _init_mediapipe(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=1,         # 0=Lite 1=Full
        )

    # ── Public: Predict on one BGR frame ──────────────────────────────────
    def predict_frame(self, bgr_frame):
        """
        Parameters
        ----------
        bgr_frame : np.ndarray  Raw BGR frame from cv2.VideoCapture.read()

        Returns
        -------
        label      : str   Predicted class name, or "" if no hand detected
        confidence : float Prediction confidence 0-100
        annotated  : np.ndarray  Frame with landmarks + bounding box drawn
        hand_bbox  : tuple (x1, y1, x2, y2) in pixel coordinates, or None
        """
        annotated = bgr_frame.copy()
        h, w = annotated.shape[:2]

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return "", 0.0, annotated, None

        # ── Pick the first detected hand ──
        hand_lm = results.multi_hand_landmarks[0]

        # Draw landmarks (fancy style)
        mp_draw.draw_landmarks(
            annotated, hand_lm,
            mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style()
        )

        # Bounding box with padding
        xs = [lm.x * w for lm in hand_lm.landmark]
        ys = [lm.y * h for lm in hand_lm.landmark]
        pad = 30
        x1 = max(0,  int(min(xs)) - pad)
        y1 = max(0,  int(min(ys)) - pad)
        x2 = min(w,  int(max(xs)) + pad)
        y2 = min(h,  int(max(ys)) + pad)

        # Ensure the crop is valid
        if x2 <= x1 or y2 <= y1:
            return "", 0.0, annotated, None

        # Crop hand region from original (clean) frame
        hand_crop = bgr_frame[y1:y2, x1:x2]
        if hand_crop.size == 0:
            return "", 0.0, annotated, None

        # Preprocess for model
        img_resized = cv2.resize(hand_crop, self.img_size)
        img_array   = img_resized.astype("float32") / 255.0
        img_batch   = np.expand_dims(img_array, axis=0)   # shape (1, H, W, 3)

        # Predict
        preds      = self.model.predict(img_batch, verbose=0)[0]
        idx        = int(np.argmax(preds))
        confidence = float(preds[idx]) * 100.0
        label      = self.classes[idx] if idx < len(self.classes) else "Unknown"

        return label, confidence, annotated, (x1, y1, x2, y2)

    # ── Public: Clean up MediaPipe resources ──────────────────────────────
    def close(self):
        if self.hands:
            self.hands.close()


# ─────────────────────────────────────────────
#  STANDALONE DEMO (optional)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick test: opens the webcam, runs predictions, and prints results.
    For the full UI experience, run main.py instead.
    """
    print("[INFO] Starting standalone prediction demo. Press Q to quit.")

    try:
        predictor = SignPredictor()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        label, conf, annotated, bbox = predictor.predict_frame(frame)

        if label:
            cv2.putText(annotated, f"{label}  ({conf:.1f}%)", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            if bbox:
                cv2.rectangle(annotated, bbox[:2], bbox[2:], (0, 255, 0), 2)

        cv2.imshow("Prediction Demo", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    predictor.close()
    cap.release()
    cv2.destroyAllWindows()
