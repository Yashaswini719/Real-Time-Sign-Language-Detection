import cv2
import os
import time
import mediapipe as mp
import numpy as np
SAVE_DIR = "collected_data"          # Root folder for dataset images
IMAGES_PER_CLASS = 200               # How many images to capture per class
IMG_SIZE = (224, 224)                # Resize images to this size before saving

# All sign classes we want to recognize
CLASSES = (
    [str(i) for i in range(10)]      # 0–9
    + [chr(c) for c in range(65, 91)]  # A–Z
    + ["Hello", "Thanks", "Yes", "No", "Please",
       "Sorry", "Help", "Stop", "ILoveYou"]
)

# ─────────────────────────────────────────────
#  MEDIAPIPE HANDS SETUP
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ─────────────────────────────────────────────
#  HELPER: Create folders for every class
# ─────────────────────────────────────────────
def create_directories():
    """Create one sub-folder per sign class inside SAVE_DIR."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    for cls in CLASSES:
        os.makedirs(os.path.join(SAVE_DIR, cls), exist_ok=True)
    print(f"[INFO] Folders created/verified inside '{SAVE_DIR}/'")


# ─────────────────────────────────────────────
#  HELPER: Draw landmarks + bounding box on frame
# ─────────────────────────────────────────────
def draw_hand_info(frame, results):
    """Overlay hand landmarks and a bounding box on *frame* (in-place)."""
    h, w, _ = frame.shape
    if results.multi_hand_landmarks:
        for hand_lm in results.multi_hand_landmarks:
            # Draw skeleton
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            # Bounding box from landmarks
            xs = [lm.x * w for lm in hand_lm.landmark]
            ys = [lm.y * h for lm in hand_lm.landmark]
            x1, y1 = max(0, int(min(xs)) - 20), max(0, int(min(ys)) - 20)
            x2, y2 = min(w, int(max(xs)) + 20), min(h, int(max(ys)) + 20)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


# ─────────────────────────────────────────────
#  MAIN COLLECTION LOOP
# ─────────────────────────────────────────────
def collect_data():
    """Interactive capture loop: press SPACE to start capturing each class."""
    create_directories()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot access webcam. Check your camera connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[INFO] Dataset collection started.")
    print(f"[INFO] {len(CLASSES)} classes × {IMAGES_PER_CLASS} images = "
          f"{len(CLASSES) * IMAGES_PER_CLASS} total images\n")

    for class_name in CLASSES:
        save_path = os.path.join(SAVE_DIR, class_name)
        img_count = 0

        print(f"[CLASS] Preparing to collect: '{class_name}'")
        print("        Show the sign to the camera, then press SPACE to start.")

        # ── Wait for user to be ready ──
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)   # Mirror for natural feel
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)
            frame = draw_hand_info(frame, res)

            # Overlay instructions
            cv2.putText(frame, f"Class: {class_name}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to start capturing | Q to quit",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Data Collection - Sign Language", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                break
            if key == ord('q'):
                print("[INFO] Collection aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return

        # ── Capture frames ──
        print(f"[INFO] Capturing {IMAGES_PER_CLASS} images for '{class_name}'…")
        while img_count < IMAGES_PER_CLASS:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)
            frame = draw_hand_info(frame, res)

            # Progress overlay
            progress = int((img_count / IMAGES_PER_CLASS) * 100)
            cv2.putText(frame, f"Class: {class_name}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Captured: {img_count}/{IMAGES_PER_CLASS}  ({progress}%)",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Progress bar
            bar_x1, bar_y = 10, 100
            bar_w = int((img_count / IMAGES_PER_CLASS) * 300)
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + 300, bar_y + 15), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + 15), (0, 200, 0), -1)

            cv2.imshow("Data Collection - Sign Language", frame)

            # Save resized image
            img_resized = cv2.resize(frame, IMG_SIZE)
            filename = os.path.join(save_path, f"{class_name}_{img_count:04d}.jpg")
            cv2.imwrite(filename, img_resized)
            img_count += 1

            # Small delay for variety (avoid duplicate frames)
            time.sleep(0.05)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Collection aborted mid-class.")
                cap.release()
                cv2.destroyAllWindows()
                return

        print(f"[DONE] Saved {img_count} images for '{class_name}' → {save_path}\n")

    cap.release()
    cv2.destroyAllWindows()
    print("=" * 50)
    print("[SUCCESS] Dataset collection complete!")
    print(f"          All images saved in '{SAVE_DIR}/'")
    print("=" * 50)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    collect_data()
