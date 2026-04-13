# Voice Dictate

A lightweight macOS dictation tool that runs in the background. Hold a keyboard shortcut, speak, release — your transcribed text is pasted into whatever app you're using.

- **Transcription:** Mistral Voxtral Mini (via Mistral API)
- **Post-processing:** Automatic paragraph formatting for emails/messages (Ministral 3B)
- **UI:** Minimal floating pill with live waveform animation
- **Platform:** macOS only (uses AppKit, tkinter, launchd)

## How it works

1. **Hold `Ctrl + Shift`** — recording starts, a small pill widget appears at the bottom of your screen
2. **Release** — audio is sent to Mistral for transcription
3. **Text is auto-pasted** into the app you were using (via clipboard + Cmd-V)

Consecutive dictations are separated by a space. Emails and messages get automatic paragraph formatting.

Quit with `Ctrl + Shift + Q`.

## Requirements

- macOS
- Python 3.13 (`brew install python@3.13`)
- tkinter (`brew install python-tk@3.13`)
- [Mistral API key](https://console.mistral.ai/)
- Microphone access (macOS will prompt on first run)
- Accessibility access for keyboard listening (System Settings → Privacy & Security → Accessibility)

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/your-user/voice-dictate.git
cd voice-dictate
pip3.13 install -r requirements.txt
pip3.13 install pyobjc-framework-Cocoa
```

### 2. Configure API key

Create a `.env` file in the project root:

```bash
echo "MISTRAL_API_KEY=your-key-here" > .env
```

### 3. Test manually

```bash
python3.13 voice_dictate.py
```

If it runs without errors, you're ready to set up auto-start.

### 4. Auto-start on login (launchd)

This sets up voice-dictate as a background service that starts on login and auto-restarts on crash.

```bash
# Copy the example plist
cp launchd/com.voice-dictate.plist.example ~/Library/LaunchAgents/com.voice-dictate.plist
```

Edit `~/Library/LaunchAgents/com.voice-dictate.plist` and replace the placeholder paths with your actual paths:

- `/path/to/python3.13` → your Python path (run `which python3.13` to find it)
- `/path/to/voice-dictate` → the directory where you cloned the repo

Then load the service:

```bash
launchctl load ~/Library/LaunchAgents/com.voice-dictate.plist
```

The tool is now running and will auto-start on every login.

### Useful commands

```bash
# Restart
launchctl kickstart -k gui/$(id -u)/com.voice-dictate

# Stop
launchctl unload ~/Library/LaunchAgents/com.voice-dictate.plist

# View logs
tail -f /tmp/voice-dictate.out
tail -f /tmp/voice-dictate.err
```

## macOS Permissions

On first run, macOS will ask for two permissions:

1. **Microphone** — needed for recording audio
2. **Accessibility** — needed for global keyboard shortcut listening and simulated keystrokes

Grant both in **System Settings → Privacy & Security**.

## Configuration

Edit the constants at the top of `voice_dictate.py`:

| Setting | Default | Description |
|---|---|---|
| `FORMAT_ENABLED` | `True` | Auto-format emails/messages with paragraph breaks |
| `FORMAT_MODEL` | `ministral-3b-latest` | LLM model for formatting |
| `FORMAT_TIMEOUT` | `2.0` | Timeout (seconds) for formatting — falls back to raw text |
| `SAMPLE_RATE` | `16000` | Audio sample rate |
| `DISCO` | `True` | Rainbow waveform bars — each bar cycles through the color spectrum while recording |

## License

MIT
