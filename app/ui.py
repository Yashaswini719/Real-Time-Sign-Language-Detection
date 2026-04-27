"""
========================================================
  Real-Time Sign Language Detection System
  FILE: app/ui.py
  PURPOSE: Renders a clean, professional overlay on top
           of the OpenCV webcam frame — shows sign label,
           confidence bar, voice status, FPS counter, and
           a semi-transparent info panel.
========================================================
"""

import cv2
import numpy as np
import time


# ─────────────────────────────────────────────
#  COLOUR PALETTE  (BGR)
# ─────────────────────────────────────────────
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,   0,   0)
C_GREEN   = (0,   220, 100)
C_CYAN    = (255, 220, 0)
C_ORANGE  = (0,   165, 255)
C_RED     = (60,  60,  220)
C_DARK    = (20,  20,  20)
C_PANEL   = (30,  30,  30)    # Panel background


class UIRenderer:
    """
    Stateless renderer: call `render(frame, ...)` every frame to get a fully
    annotated frame. All drawing is on a copy so the original stays clean.
    """

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        self.w = frame_width
        self.h = frame_height
        self._fps_times = []   # Rolling window for FPS calculation

    # ─────────────────────────────────────────
    #  PUBLIC MAIN RENDER METHOD
    # ─────────────────────────────────────────
    def render(self, frame: np.ndarray,
               label: str,
               confidence: float,
               voice_status: str,
               bbox=None,
               show_fps: bool = True) -> np.ndarray:
        """
        Draw the full UI overlay on *frame*.

        Parameters
        ----------
        frame        : BGR frame (already has MediaPipe landmarks drawn on it)
        label        : Predicted sign label, e.g. "A" or "Hello"
        confidence   : Confidence value 0-100
        voice_status : String from VoiceEngine.status
        bbox         : (x1, y1, x2, y2) bounding box pixels, or None
        show_fps     : Whether to render FPS counter

        Returns
        -------
        Annotated BGR frame (same shape as input).
        """
        out = frame.copy()

        # FPS
        if show_fps:
            fps = self._calc_fps()
            self._draw_fps(out, fps)

        if label:
            # Bounding box
            if bbox:
                self._draw_bbox(out, bbox, confidence)

            # Bottom panel
            self._draw_bottom_panel(out, label, confidence, voice_status)

        else:
            # No hand → show guidance
            self._draw_no_hand_hint(out)

        # Top banner (title bar)
        self._draw_top_banner(out)

        return out

    # ─────────────────────────────────────────
    #  PRIVATE DRAW HELPERS
    # ─────────────────────────────────────────

    def _draw_top_banner(self, frame):
        """Dark translucent top strip with project title."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.w, 45), C_PANEL, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        title = "Real-Time Sign Language Detection System"
        cv2.putText(frame, title, (10, 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, C_CYAN, 1, cv2.LINE_AA)

    def _draw_fps(self, frame, fps: float):
        """Top-right FPS badge."""
        text = f"FPS: {fps:.1f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        x = self.w - tw - 12
        cv2.putText(frame, text, (x, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GREEN, 1, cv2.LINE_AA)

    def _draw_bbox(self, frame, bbox, confidence: float):
        """Coloured bounding box around detected hand."""
        x1, y1, x2, y2 = bbox
        # Colour shifts green→orange→red with confidence (high = green)
        ratio = confidence / 100.0
        r = int(255 * (1 - ratio))
        g = int(255 * ratio)
        colour = (0, g, r)   # BGR

        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Small corner accents (decorative)
        L = 18
        T = 3
        for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                                   (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(frame, (px, py), (px + dx * L, py), colour, T)
            cv2.line(frame, (px, py), (px, py + dy * L), colour, T)

    def _draw_bottom_panel(self, frame, label: str,
                            confidence: float, voice_status: str):
        """
        Semi-transparent bottom panel showing:
          • Big sign label
          • Confidence percentage + bar
          • Voice output status
        """
        panel_h = 130
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.h - panel_h), (self.w, self.h),
                      C_PANEL, -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        base_y = self.h - panel_h + 10

        # ── Sign label (large) ──
        label_text = f"Sign: {label}"
        cv2.putText(frame, label_text,
                    (15, base_y + 42),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, C_GREEN, 2, cv2.LINE_AA)

        # ── Confidence percentage ──
        conf_text = f"Confidence: {confidence:.1f}%"
        cv2.putText(frame, conf_text,
                    (15, base_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_WHITE, 1, cv2.LINE_AA)

        # ── Confidence bar ──
        bar_x, bar_y = 15, base_y + 85
        bar_w = self.w - 30
        bar_h = 10

        # Background track
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        # Fill
        fill_w = int(bar_w * confidence / 100.0)
        if fill_w > 0:
            # Gradient-ish: green for high confidence, orange/red for low
            ratio = confidence / 100.0
            r = int(255 * (1 - ratio))
            g = int(200 * ratio)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), (0, g, r), -1)

        # ── Voice status ──
        voice_icon = "🔊" if "Speaking" in voice_status else "⏳"
        voice_text = f"Voice: {voice_status}"
        cv2.putText(frame, voice_text,
                    (15, base_y + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_CYAN, 1, cv2.LINE_AA)

    def _draw_no_hand_hint(self, frame):
        """Hint shown when no hand is detected."""
        text = "Show your hand to the camera…"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        x = (self.w - tw) // 2
        y = self.h // 2 + 20

        # Drop-shadow
        cv2.putText(frame, text, (x + 2, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_ORANGE, 2, cv2.LINE_AA)

    # ─────────────────────────────────────────
    #  FPS CALCULATOR
    # ─────────────────────────────────────────
    def _calc_fps(self) -> float:
        now = time.time()
        self._fps_times.append(now)
        # Keep only last 30 timestamps
        self._fps_times = [t for t in self._fps_times if now - t < 2.0]
        if len(self._fps_times) < 2:
            return 0.0
        elapsed = self._fps_times[-1] - self._fps_times[0]
        return (len(self._fps_times) - 1) / elapsed if elapsed > 0 else 0.0
