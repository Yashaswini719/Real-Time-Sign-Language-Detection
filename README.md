# Real-Time Sign Language Detection System with Voice Output

> **Final Year Project** | Python · OpenCV · MediaPipe · TensorFlow/Keras · pyttsx3

---

## Project Overview

This system recognises hand signs via a live webcam feed and converts them into **on-screen text** and **spoken voice output** in real time.  
It detects:

| Category | Signs |
|---|---|
| Digits | 0 – 9 |
| Alphabet | A – Z |
| Daily phrases | Hello, Thanks, Yes, No, Please, Sorry, Help, Stop, I Love You |

**Technology stack**

| Library | Role |
|---|---|
| OpenCV | Real-time video capture & frame rendering |
| MediaPipe | Hand landmark detection (21 keypoints per hand) |
| TensorFlow / Keras | CNN model training & inference |
| pyttsx3 | Offline text-to-speech voice output |
| NumPy / scikit-learn | Data processing & evaluation |
| Matplotlib | Training curve visualisation |

---

## Project Structure

```
Real-Time-Sign-Language-Detection/
│
├── dataset/            ← (placeholder – add pre-built datasets here)
├── collected_data/     ← Created by collect_data.py
│   ├── A/  B/  …  Z/
│   ├── 0/  1/  …  9/
│   └── Hello/  Thanks/  …
│
├── model/              ← Created by train_model.py
│   ├── sign_language_model.h5
│   ├── model_metadata.json
│   └── training_curves.png
│
├── app/
│   ├── __init__.py
│   ├── voice_engine.py ← Thread-safe TTS wrapper
│   └── ui.py           ← OpenCV UI overlay renderer
│
├── screenshots/        ← Created on first screenshot (S key)
│
├── main.py             ← ★ Run this to start the application
├── collect_data.py     ← Step 1: collect training images
├── train_model.py      ← Step 2: train the CNN model
├── predict.py          ← Prediction engine (also standalone demo)
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python **3.8 – 3.10** (TensorFlow 2.13 requires ≤ 3.10)
- A working webcam
- VS Code (recommended) or any terminal

### 1 – Clone / download the project

```bash
# If using git
git clone https://github.com/your-username/Real-Time-Sign-Language-Detection.git
cd Real-Time-Sign-Language-Detection
```

### 2 – Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 – Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows users:** if `pyttsx3` gives a COM error, install  
> `pip install pywin32` and restart your terminal.

---

## How to Run the Project

### Step 1 — Collect training data

```bash
python collect_data.py
```

- A webcam window opens.
- For **each class**, position your hand and press **SPACE** to start capturing.
- The script captures **200 images per class** automatically.
- Press **Q** to stop early.

### Step 2 — Train the model

```bash
python train_model.py
```

- Loads images from `collected_data/`.
- Trains a CNN for up to 30 epochs (early stopping kicks in when accuracy plateaus).
- Saves `model/sign_language_model.h5` and `model/model_metadata.json`.
- Saves `model/training_curves.png` (accuracy & loss graph).

Expected training time: **~10–30 minutes** depending on GPU/CPU.

### Step 3 — Run the application

```bash
python main.py
```

The live webcam window opens with the full UI.

---

## Keyboard Shortcuts (in the app window)

| Key | Action |
|---|---|
| `Q` or `ESC` | Quit the application |
| `V` | Toggle voice on / off |
| `S` | Save a screenshot to `screenshots/` |
| `H` | Show / hide the help panel |

---

## UI Overview

```
┌─────────────────────────────────────────┐
│ Real-Time Sign Language Detection …  FPS│  ← top banner
│                                         │
│         (live webcam + landmarks)       │
│                                         │
│                                         │
├─────────────────────────────────────────┤
│ Sign: Hello         Confidence: 92.3%   │  ← bottom panel
│ ██████████████████████░░░░░  (bar)      │
│ Voice: Ready                            │
└─────────────────────────────────────────┘
```

---

## Required Libraries

```
opencv-python   — webcam & image processing
mediapipe       — hand landmark detection
tensorflow      — deep learning framework
keras           — high-level model API
numpy           — numerical arrays
pandas          — data analysis
matplotlib      — plotting training curves
scikit-learn    — train/val split, metrics
pyttsx3         — offline text-to-speech
pyautogui       — optional GUI automation helpers
Pillow          — image utility support
```

---

## How the Model Works

1. **MediaPipe** detects the hand and locates 21 landmarks in each frame.  
2. A bounding box crops the hand region from the frame.  
3. The crop is resized to **64 × 64 px** and normalised.  
4. The **CNN** (3 Conv-BN-Pool blocks → Dense → Softmax) classifies it into one of the sign classes.  
5. If confidence ≥ 65 % and the same sign is held for 1.5 s, **pyttsx3** speaks the label aloud.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Webcam not opening | Try `WEBCAM_INDEX = 1` in `main.py` |
| `model not found` error | Run `train_model.py` first |
| Voice not working | Install `pywin32` (Windows) or `espeak` (Linux) |
| Low accuracy | Collect more varied images; adjust lighting |
| Slow FPS | Reduce `IMG_SIZE` or use a GPU |

---

## Future Improvements

- [ ] **Word / sentence builder** — accumulate detected letters into words
- [ ] **Two-hand signs** — support gestures requiring both hands
- [ ] **Transfer learning** — use MobileNetV2 / EfficientNet for higher accuracy
- [ ] **Real-time translation** — integrate Google Translate API
- [ ] **Mobile app** — deploy with TensorFlow Lite on Android / iOS
- [ ] **GUI frontend** — replace OpenCV window with a Tkinter or PyQt5 UI
- [ ] **Cloud model** — serve predictions via a REST API (FastAPI + Docker)
- [ ] **Regional sign languages** — extend to ISL, BSL, ASL variants

---



*Built with ❤️ using Python, MediaPipe, TensorFlow, and OpenCV*
