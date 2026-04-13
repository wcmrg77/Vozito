#!/usr/bin/env python3.13
"""
Voice Dictate
Shortcut: Ctrl + Shift (hold → release to transcribe)
Transcription: Mistral Voxtral Mini
"""

import os
import sys
import json
import math
import time
import queue
import tempfile
import threading
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

try:
    import tkinter as tk
except ImportError:
    print("❌ tkinter not found. Install with: brew install python-tk@3.13")
    sys.exit(1)

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
import pyperclip
from pynput import keyboard

try:
    from AppKit import (
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSApplicationActivateIgnoringOtherApps,
        NSRunningApplication,
        NSWorkspace,
    )
    HAS_APPKIT = True
except ImportError:
    HAS_APPKIT = False


# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHANNELS = 1
ENV_PATH = Path(__file__).parent / ".env"

# Post-processing (LLM reformatting of the raw transcript)
FORMAT_ENABLED = True
FORMAT_MODEL   = "ministral-3b-latest"
FORMAT_TIMEOUT = 2.0   # seconds; fall back to raw text if exceeded


# ── .env ──────────────────────────────────────────────────────────────────────

def load_env():
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

load_env()
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")


# ── State ─────────────────────────────────────────────────────────────────────

recording = False
audio_frames: list = []
lock = threading.Lock()
shift_down = False
ctrl_down = False
first_paste = True          # first transcription gets no leading space
target_pid: int = 0         # app that was frontmost when recording started
current_level: float = 0.0  # latest RMS volume from the mic (0..~0.3)
level_lock = threading.Lock()
ui_queue: queue.Queue = queue.Queue()


# ── macOS focus helpers ──────────────────────────────────────────────────────

def hide_from_dock():
    """Make Python an accessory app (no dock icon, no Cmd-Tab entry)."""
    if HAS_APPKIT:
        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory
        )


def capture_frontmost_pid():
    """Remember which app was frontmost right before we show the widget."""
    global target_pid
    if not HAS_APPKIT:
        target_pid = 0
        return
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    target_pid = app.processIdentifier() if app else 0


def activate_target_app():
    """Bring the captured app back to the front so Cmd-V lands there."""
    if not HAS_APPKIT or not target_pid:
        return
    app = NSRunningApplication.runningApplicationWithProcessIdentifier_(target_pid)
    if app:
        app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe_audio(audio_path: Path) -> str:
    boundary = "----VoxtralBoundary9876543210"
    file_bytes = audio_path.read_bytes()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{audio_path.name}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + file_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"voxtral-mini-latest"
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="language"\r\n\r\n'
        f"de"
        f"\r\n--{boundary}--\r\n"
    ).encode()

    req = urllib.request.Request(
        "https://api.mistral.ai/v1/audio/transcriptions",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result.get("text", "").strip()
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"❌ API {e.code}: {body}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"❌ {e}", file=sys.stderr)
        return ""


# ── Formatting ────────────────────────────────────────────────────────────────

FORMAT_SYSTEM_PROMPT = """Du bist ein Formatierungs-Assistent für Diktate.

Deine einzige Aufgabe: füge sinnvolle Absätze und Zeilenumbrüche ein, \
wenn der diktierte Text eine Nachricht oder E-Mail ist \
(erkennbar an Anrede wie "Hi", "Hallo", "Sehr geehrte", "Liebe/r", "Guten Tag", "Moin", \
und/oder Verabschiedung wie "Viele Grüße", "Liebe Grüße", "MfG", "Mit freundlichen Grüßen", \
"Beste Grüße", "Cheers", "Best", "Regards").

Strikte Regeln — Verstöße sind nicht erlaubt:
- Du darfst NUR Whitespace verändern (Leerzeichen, Zeilenumbrüche).
- Ändere KEIN Wort, KEINEN Buchstaben, KEIN Komma, KEINEN Punkt.
- Korrigiere KEINE Grammatik, Rechtschreibung oder Zeichensetzung.
- Übersetze NICHT.
- Wenn der Text keine Nachricht/E-Mail ist (Notiz, Stichpunkt, Gedanke, Frage, Code, …), \
gib ihn exakt unverändert zurück.
- Gib NUR den formatierten Text zurück — keine Erklärungen, keine Anführungszeichen, kein Präfix."""

# Few-shot examples baked into the request. Covers: formal DE email (no
# trailing punct on name), casual DE email with trailing period on signature,
# pure DE note (must stay unchanged), EN email with comma-separated sign-off.
_FORMAT_FEWSHOT = [
    ("Hallo Max, wie geht es dir? Ich wollte kurz Bescheid geben, "
     "dass das Meeting morgen um 10 Uhr stattfindet. Viele Grüße Gregor",
     "Hallo Max,\n\nwie geht es dir? Ich wollte kurz Bescheid geben, "
     "dass das Meeting morgen um 10 Uhr stattfindet.\n\nViele Grüße\nGregor"),
    ("Hi Vincent, wollte mal fragen, wie es dir so geht. Danke für die "
     "Nachricht und ich melde mich die nächsten Tage. Beste Grüße, Gregor.",
     "Hi Vincent,\n\nwollte mal fragen, wie es dir so geht. Danke für die "
     "Nachricht und ich melde mich die nächsten Tage.\n\nBeste Grüße,\nGregor."),
    ("das ist nur eine kurze notiz, nichts besonderes.",
     "das ist nur eine kurze notiz, nichts besonderes."),
    ("Hi Sarah, just a quick heads-up that the deploy is done and everything "
     "looks green on staging. Let me know if you see anything weird. Cheers, Sarah.",
     "Hi Sarah,\n\njust a quick heads-up that the deploy is done and everything "
     "looks green on staging. Let me know if you see anything weird.\n\nCheers,\nSarah."),
]


def _strip_ws(s: str) -> str:
    """Collapse to non-whitespace characters for the sanity check."""
    return "".join(s.split())


_TRAIL_PUNCT = set(".!?,;:")


def _reattach_trailing_punct(raw: str, formatted: str) -> str | None:
    """
    If `formatted` is `raw` missing ONLY trailing punctuation at the very end
    (e.g. the LLM dropped the final "." from "Gregor."), return a patched
    version with that punctuation re-appended. Otherwise return None.
    """
    raw_ws = _strip_ws(raw)
    fmt_ws = _strip_ws(formatted)
    if not raw_ws.startswith(fmt_ws) or raw_ws == fmt_ws:
        return None
    missing = raw_ws[len(fmt_ws):]
    if not all(c in _TRAIL_PUNCT for c in missing):
        return None
    return formatted.rstrip() + missing


def format_transcript(raw: str) -> str:
    """Ask Mistral to add paragraph structure; fall back to raw on any issue."""
    if not FORMAT_ENABLED or not raw.strip():
        return raw

    messages = [{"role": "system", "content": FORMAT_SYSTEM_PROMPT}]
    for user_msg, assistant_msg in _FORMAT_FEWSHOT:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": raw})

    body = json.dumps({
        "model": FORMAT_MODEL,
        "temperature": 0.0,
        "max_tokens": 800,
        "messages": messages,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.mistral.ai/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=FORMAT_TIMEOUT) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        formatted = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️  format skipped ({e}) — using raw", file=sys.stderr)
        return raw

    # Safety net: non-whitespace characters must match exactly.
    # Tolerated exception: the LLM dropped trailing punctuation at the very
    # end (common failure mode on name signatures) — we just glue it back.
    if _strip_ws(formatted) == _strip_ws(raw):
        return formatted

    patched = _reattach_trailing_punct(raw, formatted)
    if patched is not None:
        return patched

    print("⚠️  format changed non-whitespace — using raw", file=sys.stderr)
    return raw


# ── Paste ─────────────────────────────────────────────────────────────────────

def paste_text(text: str):
    global first_paste
    needs_space = not first_paste
    first_paste = False

    # Clipboard always holds the clean text, so manual pastes into the
    # correct field work too — no leading-space artifact.
    pyperclip.copy(text)
    activate_target_app()   # return focus to the app the user was in
    time.sleep(0.15)

    if needs_space:
        # Separator between consecutive dictations is typed, not clipboarded.
        subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to keystroke " "'],
            check=False,
            stderr=subprocess.DEVNULL,
        )

    subprocess.run(
        ["osascript", "-e",
         'tell application "System Events" to keystroke "v" using {command down}'],
        check=False,
        stderr=subprocess.DEVNULL,
    )


# ── Recording ─────────────────────────────────────────────────────────────────

def start_recording():
    global recording, audio_frames
    with lock:
        if recording:
            return
        capture_frontmost_pid()   # before the widget can steal focus
        recording = True
        audio_frames = []
    ui_queue.put("recording")
    print("🎙️  Recording...")


def stop_and_transcribe():
    global recording, audio_frames
    with lock:
        if not recording:
            return
        recording = False
        frames = list(audio_frames)

    if not frames:
        ui_queue.put("hide")
        return

    ui_queue.put("transcribing")
    audio_data = np.concatenate(frames, axis=0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        wavfile.write(tmp_path, SAMPLE_RATE, audio_data)
        text = transcribe_audio(tmp_path)
        if text:
            text = format_transcript(text)
            print(f"✅ {text}")
            paste_text(text)
        else:
            print("⚠️  Empty transcript.")
    finally:
        tmp_path.unlink(missing_ok=True)
        ui_queue.put("hide")


def audio_callback(indata, frames, time_info, status):
    global current_level
    if recording:
        audio_frames.append(indata.copy())
        # RMS volume for the visualizer, normalized to int16 range
        samples = indata.astype(np.float32)
        rms = float(np.sqrt(np.mean(samples * samples))) / 32768.0
        with level_lock:
            current_level = rms


# ── Keyboard ──────────────────────────────────────────────────────────────────

def _is_char(key, ch: str) -> bool:
    """True if `key` is the given literal character (case-insensitive)."""
    return (
        hasattr(key, "char")
        and key.char is not None
        and key.char.lower() == ch.lower()
    )


def on_press(key):
    global shift_down, ctrl_down, recording, audio_frames
    if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
        shift_down = True
    elif key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        ctrl_down = True
    elif shift_down and ctrl_down and _is_char(key, "q"):
        # Quit combo: Ctrl + Shift + Q. Cancel any in-flight recording first.
        with lock:
            recording = False
            audio_frames = []
        ui_queue.put("hide")
        ui_queue.put("quit")
        return False

    if shift_down and ctrl_down:
        start_recording()


def on_release(key):
    global shift_down, ctrl_down
    was_recording = recording

    if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
        shift_down = False
    elif key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        ctrl_down = False

    if was_recording and (not shift_down or not ctrl_down):
        threading.Thread(target=stop_and_transcribe, daemon=True).start()


# ── Widget ────────────────────────────────────────────────────────────────────

class RecordingWidget:
    # Pill geometry
    W = 104         # pill width
    H = 34          # pill height (== 2*R → fully rounded ends)
    R = 17          # corner radius

    # Bar geometry
    N_BARS  = 20
    BAR_W   = 2
    BAR_GAP = 2
    BAR_MIN = 3
    BAR_MAX = 22    # peak bar height

    FRAME_MS = 55   # animation frame interval
    GAIN     = 10.0 # RMS → visual level amplification (tune for your mic)

    WHITE = "#ffffff"
    FILL  = "#1a1a1a"   # pill background

    def __init__(self):
        self.root = tk.Tk()
        # Must run *after* tk.Tk() so we flip the policy on Tk's own
        # TKApplication subclass (not a bare NSApplication created by us).
        hide_from_dock()

        self.root.wm_overrideredirect(True)
        self.root.wm_attributes("-topmost", True)
        self.root.wm_attributes("-transparent", True)
        self.root.configure(bg="systemTransparent")

        self.canvas = tk.Canvas(
            self.root,
            width=self.W, height=self.H,
            bg="systemTransparent",
            highlightthickness=0, borderwidth=0,
        )
        self.canvas.pack()

        # Cached bar x-positions (centered horizontally)
        total_w = self.N_BARS * self.BAR_W + (self.N_BARS - 1) * self.BAR_GAP
        start_x = (self.W - total_w) // 2
        self._bar_x = [start_x + i * (self.BAR_W + self.BAR_GAP) for i in range(self.N_BARS)]

        self._draw_pill()
        self._bars: list = []
        self._init_bars()
        self._animating = False
        self._history = [0.0] * self.N_BARS       # rolling bar-level buffer

        # Position: bottom-center, 32 px above dock
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{self.W}x{self.H}+{(sw - self.W) // 2}+{sh - self.H - 32}")
        # Map the window once so macOS applies wm_overrideredirect properly,
        # then keep it mapped but invisible via alpha for runtime show/hide
        # (avoids Space-switching that withdraw/deiconify triggers).
        self.root.withdraw()
        self.root.deiconify()
        self.root.wm_attributes("-topmost", True)
        self.root.lift()
        self.root.wm_attributes("-alpha", 0.0)

        self.root.after(80, self._poll)

    # ── drawing helpers ───────────────────────────────────────────────────────

    def _draw_pill(self):
        r = self.R
        x1, y1, x2, y2 = 0, 0, self.W, self.H
        points = [
            x1 + r, y1,  x2 - r, y1,
            x2,     y1,  x2,     y1 + r,
            x2,     y2 - r, x2,  y2,
            x2 - r, y2,  x1 + r, y2,
            x1,     y2,  x1,     y2 - r,
            x1,     y1 + r, x1,  y1,
        ]
        self.canvas.create_polygon(
            points, smooth=True, splinesteps=36,
            fill=self.FILL, outline="",
        )

    def _set_bar(self, bar, x, h):
        cy = self.H / 2
        self.canvas.coords(bar, x, cy - h / 2, x + self.BAR_W, cy + h / 2)

    def _init_bars(self):
        self._bars = []
        for x in self._bar_x:
            bar = self.canvas.create_rectangle(
                0, 0, 0, 0, fill=self.WHITE, outline="",
            )
            self._set_bar(bar, x, self.BAR_MIN)
            self._bars.append(bar)

    def _reset_bars(self):
        for bar, x in zip(self._bars, self._bar_x):
            self._set_bar(bar, x, self.BAR_MIN)

    # ── animation ─────────────────────────────────────────────────────────────

    def _animate(self):
        if not self._animating:
            return
        span = self.BAR_MAX - self.BAR_MIN

        with level_lock:
            rms = current_level
        # amplify + sqrt-compress for visual dynamic range, clip to [0, 1]
        level = min(1.0, math.sqrt(max(0.0, rms) * self.GAIN))
        # shift history left, newest value enters on the right
        self._history = self._history[1:] + [level]
        for i, (bar, x) in enumerate(zip(self._bars, self._bar_x)):
            h = self.BAR_MIN + span * self._history[i]
            self._set_bar(bar, x, h)

        self.root.after(self.FRAME_MS, self._animate)

    # ── queue poll ────────────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                msg = ui_queue.get_nowait()
                if msg == "recording":
                    self._reset_bars()
                    self._history = [0.0] * self.N_BARS
                    self._animating = True
                    self.root.wm_attributes("-alpha", 1.0)
                    self.root.lift()
                    self._animate()
                elif msg == "transcribing":
                    # freeze bars at idle while the API call runs
                    self._animating = False
                    self._reset_bars()
                elif msg == "hide":
                    self._animating = False
                    self.root.wm_attributes("-alpha", 0.0)
                elif msg == "quit":
                    self.root.quit()
                    return
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def run(self):
        self.root.mainloop()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not MISTRAL_API_KEY:
        print("❌ MISTRAL_API_KEY not set in .env")
        sys.exit(1)

    print("🎤 Voice Dictate — Mistral Voxtral Mini")
    print("Shortcut: Ctrl + Shift (hold → release = transcribe)")
    print("Ctrl + Shift + Q to quit\n")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=audio_callback,
    )
    stream.start()

    def _run_listener():
        with keyboard.Listener(on_press=on_press, on_release=on_release) as lst:
            lst.join()

    threading.Thread(target=_run_listener, daemon=True).start()

    widget = RecordingWidget()
    widget.run()

    stream.stop()
