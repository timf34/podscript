# Podscript

*Podcast → Transcript!*

Transcribe any podcast episode or YouTube video from the command line. Generates clean markdown with speaker diarization and timestamps.

<div align="center">
    <img src="./assets/image.png" alt="Podscript - Podcast to Transcript" width="600" />
</div>


## Installation

```bash
pip install podscript
podscript --setup  # paste your ElevenLabs API key
```

You'll need an [ElevenLabs API key](https://elevenlabs.io/app/settings/api-keys).

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

## License

MIT
