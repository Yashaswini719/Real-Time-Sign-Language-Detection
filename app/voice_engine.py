"""
========================================================
  Real-Time Sign Language Detection System
  FILE: app/voice_engine.py
  PURPOSE: Thread-safe text-to-speech wrapper using pyttsx3.
           Prevents the same sign from being spoken
           repeatedly; uses a background thread so speech
           never blocks the video loop.
========================================================
"""

import pyttsx3
import threading
import time


class VoiceEngine:
    """
    Speaks detected signs aloud in a background thread.

    Key features:
    - Non-blocking: speech runs in a daemon thread.
    - Debounce: the same sign must be held for `hold_seconds`
      before it is spoken (avoids rapid-fire repetition).
    - Cool-down: after speaking, a `cooldown_seconds` silence
      period prevents the engine from triggering again immediately.
    """

    def __init__(self, hold_seconds: float = 1.5,
                 cooldown_seconds: float = 3.0):
        """
        Parameters
        ----------
        hold_seconds    : Seconds a sign must stay stable before TTS fires.
        cooldown_seconds: Seconds to wait after speaking before speaking again.
        """
        self.hold_seconds    = hold_seconds
        self.cooldown_seconds = cooldown_seconds

        # Internal state
        self._current_sign   = ""       # sign being tracked
        self._sign_start     = 0.0      # when the current sign was first seen
        self._last_spoken    = ""       # what was last spoken
        self._last_speak_time = 0.0     # when we last spoke
        self._is_speaking    = False    # True while TTS is running
        self.status          = "Ready"  # shown in the UI

        # Initialise pyttsx3 engine
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate',   150)   # words per minute
            self._engine.setProperty('volume', 1.0)   # 0.0 – 1.0
            self._available = True
        except Exception as e:
            print(f"[VOICE] pyttsx3 init failed: {e}")
            self._available = False

    # ── Public: call every frame with the current prediction ──────────────
    def update(self, sign: str, confidence: float,
               confidence_threshold: float = 70.0):
        """
        Feed the latest prediction. Speech fires when:
          1. Confidence ≥ threshold
          2. Same sign held for `hold_seconds`
          3. Not currently speaking
          4. Cool-down has elapsed (or it is a different sign)

        Parameters
        ----------
        sign                 : Predicted class label (empty string = no hand)
        confidence           : Prediction confidence 0-100
        confidence_threshold : Minimum confidence to consider speaking
        """
        if not self._available or not sign:
            if not sign:
                self.status = "No hand detected"
            return

        if confidence < confidence_threshold:
            self._current_sign = ""
            self.status = "Low confidence"
            return

        now = time.time()

        # Track how long this sign has been stable
        if sign != self._current_sign:
            self._current_sign = sign
            self._sign_start   = now

        held_duration = now - self._sign_start

        # Decide whether to speak
        cool_down_ok = (now - self._last_speak_time) >= self.cooldown_seconds
        new_sign     = (sign != self._last_spoken)

        if held_duration >= self.hold_seconds and not self._is_speaking:
            if new_sign or cool_down_ok:
                self._speak(sign)
                self._last_spoken    = sign
                self._last_speak_time = now
            else:
                remaining = int(self.cooldown_seconds - (now - self._last_speak_time))
                self.status = f"Cool-down ({remaining}s)"
        else:
            remaining_hold = max(0.0, self.hold_seconds - held_duration)
            self.status = f"Holding… ({remaining_hold:.1f}s)"

    # ── Private: spawn speech in a background thread ──────────────────────
    def _speak(self, text: str):
        def _run():
            self._is_speaking = True
            self.status = f"Speaking: {text}"
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate',   150)
                engine.setProperty('volume', 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[VOICE] Speech error: {e}")
            finally:
                self._is_speaking = False
                self.status = "Ready"

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ── Public: force-speak any text immediately ──────────────────────────
    def speak_now(self, text: str):
        """Speak `text` immediately, bypassing debounce/cool-down."""
        if self._available:
            self._speak(text)

    # ── Public: is voice available? ───────────────────────────────────────
    @property
    def available(self) -> bool:
        return self._available

    # ── Public: clean up ──────────────────────────────────────────────────
    def shutdown(self):
        """Call when the application exits."""
        pass   # pyttsx3 threads are daemons; they die with the process
