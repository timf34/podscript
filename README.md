# Podscript

*Podcast → Transcript!*

Transcribe any podcast episode or YouTube video from the command line. Generates clean markdown with speaker diarization and timestamps. Works with the [ElevenLabs API](https://elevenlabs.io/) or **fully locally** using [Whisper](https://github.com/SYSTRAN/faster-whisper) - no API key required.

<div align="center">
    <img src="./assets/image.png" alt="Podscript - Podcast to Transcript" width="600" />
</div>


## Installation

```bash
# Local transcription (free, no API key needed)
pip install podscript[local]

# Or use ElevenLabs API
pip install podscript
podscript --setup  # paste your ElevenLabs API key
```

For local mode, just add `--local` to any command. For ElevenLabs, you'll need an [API key](https://elevenlabs.io/app/settings/api-keys).

For YouTube support, also install [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [ffmpeg](https://ffmpeg.org/).

## Usage

```bash
# Transcribe a podcast from an Apple Podcasts link
podscript "https://podcasts.apple.com/us/podcast/huberman-lab/id1545953110?i=1000690"

# Transcribe a YouTube video
podscript "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Use an RSS feed directly
podscript https://feeds.simplecast.com/JGE3yC0V

# Browse episodes first
podscript https://feeds.simplecast.com/JGE3yC0V --list

# Search for a specific episode
podscript https://feeds.simplecast.com/JGE3yC0V --search "AI"

# Pick episode #3 from the list
podscript https://feeds.simplecast.com/JGE3yC0V --episode 3

# Custom output filename
podscript https://feeds.simplecast.com/JGE3yC0V --latest --output transcript.md
```

Without any flags, the default behavior is to transcribe the most recent episode.

## Output

Generates a markdown file with speaker labels and timestamps:

```markdown
# The Economics of Carbon Removal

**Podcast:** a16z Podcast
**Date:** 2/10/2026
**Duration:** 1:04:23

---

## Speaker 1
[0:00] Welcome back to the show. Today we're talking about...

## Speaker 2
[0:15] Thanks for having me. So the key challenge with carbon removal is...

## Speaker 1
[2:41] That's fascinating. How does the economics actually work at scale?
```

## Local Transcription

You can transcribe entirely offline using a local Whisper model — no API key required:

```bash
pip install podscript[local]
```

This installs `faster-whisper`, `pyannote.audio`, and `torch`.

### Usage

```bash
# Basic local transcription (uses "base" model, no speaker diarization)
podscript "https://www.youtube.com/watch?v=..." --local

# Use a larger model for better accuracy
podscript "https://www.youtube.com/watch?v=..." --local --model medium

# Enable speaker diarization with a HuggingFace token
podscript "https://feeds.example.com/rss" --local --hf-token hf_xxxxx

# Or set the token as an environment variable once
export HF_TOKEN=hf_xxxxx
podscript "https://feeds.example.com/rss" --local
```

### Model Sizes

| Model | Speed | Quality | VRAM |
|-------|-------|---------|------|
| `tiny` | Fastest | Lower | ~1 GB |
| `base` | Fast | Good (default) | ~1 GB |
| `small` | Moderate | Better | ~2 GB |
| `medium` | Slower | Great | ~5 GB |
| `large-v2` | Slowest | Best | ~10 GB |
| `large-v3` | Slowest | Best | ~10 GB |

CPU mode uses `int8` quantization automatically. GPU (CUDA) uses `float16`.

### Speaker Diarization

Speaker diarization (identifying who said what) requires a free [HuggingFace](https://huggingface.co/) token:

1. Create an account at [huggingface.co](https://huggingface.co/join)
2. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Pass it via `--hf-token` or set `HF_TOKEN` in your environment

Without a token, all speech is attributed to "Speaker 1" — still useful for single-speaker content.

## License

MIT
