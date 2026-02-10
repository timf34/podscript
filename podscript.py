#!/usr/bin/env python3
"""Podscript - Transcribe podcasts and YouTube videos using ElevenLabs Scribe API."""

import argparse
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from urllib.parse import urlparse, parse_qs

import feedparser
import requests
from dotenv import load_dotenv

# Constants
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/speech-to-text"
PAUSE_THRESHOLD = 1.0  # seconds - gap that triggers new segment


# Dataclasses
@dataclass
class Episode:
    title: str
    audio_url: str
    publish_date: str
    description: str
    duration: str


@dataclass
class TranscriptSegment:
    speaker: str
    text: str
    start: float
    end: float


# ── Helpers ──────────────────────────────────────────────────────────────────


def format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def sanitize_filename(name: str) -> str:
    """Strip non-alphanumeric chars, collapse whitespace to hyphens, max 80 chars."""
    name = re.sub(r"[^a-zA-Z0-9\s-]", "", name)
    name = re.sub(r"\s+", "-", name).strip("-")
    return name[:80]


def clean_html(text: str) -> str:
    """Strip HTML tags and unescape entities."""
    text = re.sub(r"<[^>]*>", "", text)
    return html.unescape(text).strip()


def normalize_for_matching(text: str) -> str:
    """Normalize a string for fuzzy title comparison.

    Lowercases, normalizes unicode (smart quotes/dashes → ascii equivalents),
    strips punctuation, and collapses whitespace.
    """
    text = unicodedata.normalize("NFKD", text).lower()
    # Replace common unicode punctuation with ascii
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Strip all punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_duration(duration) -> str:
    """Normalise an itunes:duration value (seconds int, HH:MM:SS string, etc.)."""
    if not duration:
        return ""
    if isinstance(duration, (int, float)):
        return format_timestamp(duration)
    duration = str(duration)
    if ":" in duration:
        return duration
    try:
        return format_timestamp(int(duration))
    except ValueError:
        return duration


# ── YouTube ──────────────────────────────────────────────────────────────────


def is_youtube_url(url: str) -> bool:
    return bool(re.search(r"(?:youtube\.com/watch|youtu\.be/|youtube\.com/shorts/)", url))


def get_yt_dlp_cmd() -> str:
    """Return the yt-dlp invocation that works on this system."""
    for cmd in ["yt-dlp", "python -m yt_dlp"]:
        try:
            subprocess.run(
                [*cmd.split(), "--version"],
                capture_output=True,
                check=True,
            )
            return cmd
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    print(
        "Error: yt-dlp is not installed. Install it with: pip install yt-dlp\n"
        "You also need ffmpeg installed for audio extraction.",
        file=sys.stderr,
    )
    sys.exit(1)


def clean_youtube_url(url: str) -> str:
    """Strip timestamp parameters that can cause shell issues."""
    try:
        from urllib.parse import urlparse, urlencode, parse_qs

        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        params.pop("t", None)
        clean_query = urlencode(params, doseq=True)
        return parsed._replace(query=clean_query).geturl()
    except Exception:
        return url


def download_youtube_audio(url: str) -> dict:
    """Download audio from YouTube, return dict with title, channel, duration, audio_path."""
    yt_dlp = get_yt_dlp_cmd()
    clean_url = clean_youtube_url(url)

    # Get metadata
    print("Fetching video info...\n")
    result = subprocess.run(
        [*yt_dlp.split(), "--no-download", "--dump-json", clean_url],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"Error fetching video info: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    info = json.loads(result.stdout)
    title = info.get("title", "Untitled")
    channel = info.get("channel") or info.get("uploader") or "Unknown"
    duration = info.get("duration", 0)

    print(f"Title: {title}")
    print(f"Channel: {channel}")
    print(f"Duration: {format_duration(duration)}\n")

    # Download audio
    temp_base = os.path.join(tempfile.gettempdir(), f"podscript-{os.getpid()}")
    temp_output = f"{temp_base}.%(ext)s"

    print("Downloading audio...")
    subprocess.run(
        [*yt_dlp.split(), "-x", "--audio-format", "mp3", "--audio-quality", "0", "-o", temp_output, clean_url],
        timeout=600,
        check=True,
    )

    audio_path = f"{temp_base}.mp3"
    if not os.path.exists(audio_path):
        print(f"Error: Audio download failed - expected file at {audio_path}", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"Downloaded: {size_mb:.1f} MB\n")

    return {"title": title, "channel": channel, "duration": duration, "audio_path": audio_path}


# ── Apple Podcasts ────────────────────────────────────────────────────────


def parse_apple_podcasts_url(url: str) -> tuple[str, str | None] | None:
    """
    If url is an Apple Podcasts link, return (podcast_id, episode_id or None).
    Otherwise return None.
    """
    parsed = urlparse(url)
    if not parsed.hostname or "podcasts.apple.com" not in parsed.hostname:
        return None
    # Extract podcast ID from path like /gb/podcast/some-name/id842818711
    m = re.search(r"/id(\d+)", parsed.path)
    if not m:
        return None
    podcast_id = m.group(1)
    # Extract episode ID from query param ?i=1000588160381
    params = parse_qs(parsed.query)
    episode_id = params.get("i", [None])[0]
    return podcast_id, episode_id


def resolve_apple_podcasts_url(url: str) -> tuple[str, str | None]:
    """
    Resolve an Apple Podcasts URL to its RSS feed URL.
    Returns (feed_url, episode_id or None).
    """
    result = parse_apple_podcasts_url(url)
    if result is None:
        raise ValueError("Not an Apple Podcasts URL")
    podcast_id, episode_id = result

    print(f"Resolving Apple Podcasts ID {podcast_id} to RSS feed...")
    resp = requests.get(
        f"https://itunes.apple.com/lookup?id={podcast_id}&entity=podcast",
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"No podcast found for Apple Podcasts ID {podcast_id}")
    feed_url = results[0].get("feedUrl")
    if not feed_url:
        raise RuntimeError(f"No RSS feed URL found for Apple Podcasts ID {podcast_id}")
    print(f"Found RSS feed: {feed_url}\n")
    return feed_url, episode_id


def scrape_apple_episode_info(apple_url: str) -> dict | None:
    """
    Scrape episode title and audio URL from an Apple Podcasts page.
    Returns dict with 'title' and optionally 'audio_url', or None on failure.
    """
    try:
        resp = requests.get(apple_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        page = resp.text
    except Exception:
        return None

    info = {}

    # Extract title — try og:title first
    m = re.search(r'<meta\s[^>]*property="og:title"\s[^>]*content="([^"]*)"', page)
    if not m:
        m = re.search(r'<meta\s[^>]*content="([^"]*)"\s[^>]*property="og:title"', page)
    if m:
        info["title"] = html.unescape(m.group(1))
    else:
        # Fallback: extract from <title> tag (often "Episode Title - Podcast Name")
        m = re.search(r"<title>([^<]+)</title>", page)
        if m:
            raw_title = html.unescape(m.group(1)).strip()
            # Apple title format is often "Episode Title - Podcast Name on Apple Podcasts"
            raw_title = re.sub(r"\s+on\s+Apple\s+Podcasts\s*$", "", raw_title)
            # Take the part before the last " - " as the episode title
            if " - " in raw_title:
                info["title"] = raw_title.rsplit(" - ", 1)[0].strip()
            else:
                info["title"] = raw_title

    # Fallback: extract a fuzzy title from the URL slug
    slug_m = re.search(r"/podcast/([^/]+)/id", apple_url)
    if slug_m and "title" not in info:
        info["title"] = slug_m.group(1).replace("-", " ")

    # Extract audio URL from streamUrl in embedded JSON
    m = re.search(r'"streamUrl"\s*:\s*"(https?://[^"]+\.mp3[^"]*)"', page)
    if not m:
        # Fallback: streamUrl without .mp3 extension (some episodes use different formats)
        m = re.search(r'"streamUrl"\s*:\s*"(https?://[^"]+)"', page)
    if m:
        info["audio_url"] = m.group(1)

    return info if info else None


# ── RSS ──────────────────────────────────────────────────────────────────────


def parse_feed(feed_url: str) -> tuple[str, list[Episode]]:
    """Parse an RSS feed, return (podcast_name, episodes)."""
    feed = feedparser.parse(feed_url)

    if feed.bozo and not feed.entries:
        raise RuntimeError(f"Failed to parse feed: {feed.bozo_exception}")

    podcast_name = getattr(feed.feed, "title", "Unknown Podcast")
    episodes: list[Episode] = []

    for entry in feed.entries:
        # Find audio URL from enclosures
        audio_url = ""
        for enc in getattr(entry, "enclosures", []):
            href = enc.get("href") or enc.get("url", "")
            if href:
                audio_url = href
                break
        if not audio_url:
            continue

        title = getattr(entry, "title", "Untitled Episode")
        pub_date = getattr(entry, "published", "")
        summary = getattr(entry, "summary", "") or getattr(entry, "description", "")
        description = clean_html(summary)[:200]
        if len(clean_html(summary)) > 200:
            description += "..."
        raw_duration = getattr(entry, "itunes_duration", "")
        duration = format_duration(raw_duration)

        episodes.append(Episode(
            title=title,
            audio_url=audio_url,
            publish_date=pub_date,
            description=description,
            duration=duration,
        ))

    return podcast_name, episodes


# ── ElevenLabs transcription ────────────────────────────────────────────────


def transcribe(source: str, *, is_file: bool = False) -> tuple[list[TranscriptSegment], float]:
    """
    Transcribe audio via ElevenLabs Scribe API.

    Args:
        source: Either a cloud URL or a local file path.
        is_file: True if source is a local file path.

    Returns:
        (segments, duration_seconds)
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY not found in environment.", file=sys.stderr)
        sys.exit(1)

    data = {
        "model_id": "scribe_v1",
        "diarize": "true",
        "timestamps_granularity": "word",
    }
    files = None

    if is_file:
        files = {"file": ("audio.mp3", open(source, "rb"), "audio/mpeg")}
    else:
        data["cloud_storage_url"] = source

    resp = requests.post(
        ELEVENLABS_API_URL,
        headers={"xi-api-key": api_key},
        data=data,
        files=files,
        timeout=600,
    )

    # Close the file handle if we opened one
    if files and "file" in files:
        files["file"][1].close()

    if resp.status_code != 200:
        print(f"ElevenLabs API error ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)

    body = resp.json()
    words = body.get("words", [])
    segments = group_into_segments(words)
    duration = words[-1]["end"] if words else 0.0

    return segments, duration


def group_into_segments(words: list[dict]) -> list[TranscriptSegment]:
    """Group words into segments by speaker changes and pauses > PAUSE_THRESHOLD."""
    if not words:
        return []

    segments: list[TranscriptSegment] = []
    cur: TranscriptSegment | None = None

    for w in words:
        speaker = w.get("speaker_id") or "speaker_0"
        start_new = (
            cur is None
            or cur.speaker != speaker
            or (w["start"] - cur.end > PAUSE_THRESHOLD)
        )

        if start_new:
            if cur is not None:
                segments.append(cur)
            cur = TranscriptSegment(speaker=speaker, text=w["text"], start=w["start"], end=w["end"])
        else:
            cur.text += " " + w["text"]
            cur.end = w["end"]

    if cur is not None:
        segments.append(cur)

    # Clean up whitespace
    for seg in segments:
        seg.text = re.sub(r"\s+", " ", seg.text).strip()

    return segments


# ── Markdown generation ─────────────────────────────────────────────────────


def speaker_name(speaker_id: str) -> str:
    """Convert speaker_0 → Speaker 1, etc."""
    m = re.match(r"speaker_(\d+)", speaker_id)
    if m:
        return f"Speaker {int(m.group(1)) + 1}"
    return speaker_id


def generate_markdown(
    title: str,
    podcast_name: str,
    segments: list[TranscriptSegment],
    duration_secs: float,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"**Podcast:** {podcast_name}",
        f"**Date:** {datetime.now().month}/{datetime.now().day}/{datetime.now().year}",
        f"**Duration:** {format_timestamp(duration_secs)}",
        "",
        "---",
        "",
    ]

    current_speaker = ""
    for seg in segments:
        name = speaker_name(seg.speaker)
        if name != current_speaker:
            lines.append(f"## {name}")
            current_speaker = name
        lines.append(f"[{format_timestamp(seg.start)}] {seg.text}")
        lines.append("")

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────


CONFIG_DIR = Path.home() / ".config" / "podscript"
CONFIG_FILE = CONFIG_DIR / ".env"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="podscript",
        description="Transcribe podcasts and YouTube videos using ElevenLabs Scribe API.",
    )
    parser.add_argument("url", nargs="?", help="RSS feed URL, YouTube URL, or Apple Podcasts URL")
    parser.add_argument("--setup", action="store_true", help="Save your ElevenLabs API key")
    parser.add_argument("--episode", type=int, metavar="N", help="Transcribe episode N (1 = most recent)")
    parser.add_argument("--search", metavar="QUERY", help="Search episodes by title/description")
    parser.add_argument("--latest", action="store_true", help="Transcribe the most recent episode (default)")
    parser.add_argument("--list", action="store_true", dest="list_episodes", help="List episodes without transcribing")
    parser.add_argument("--output", metavar="FILE", help="Output filename (default: auto-generated)")
    return parser


def setup_api_key():
    """Prompt the user for their ElevenLabs API key and save it."""
    print("Enter your ElevenLabs API key (from https://elevenlabs.io/app/settings/api-keys):")
    try:
        key = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        sys.exit(1)
    if not key:
        print("No key entered. Setup cancelled.", file=sys.stderr)
        sys.exit(1)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(f"ELEVENLABS_API_KEY={key}\n", encoding="utf-8")
    print(f"\nAPI key saved to {CONFIG_FILE}")
    print("You're all set! Try: podscript <url>")


def require_api_key():
    """Exit if ELEVENLABS_API_KEY is not set."""
    if not os.environ.get("ELEVENLABS_API_KEY", ""):
        print("Error: ELEVENLABS_API_KEY not set.", file=sys.stderr)
        print("Run `podscript --setup` to save your API key.", file=sys.stderr)
        sys.exit(1)


def main():
    # Load API key from config file, then local .env, then environment
    load_dotenv(CONFIG_FILE)
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    if args.setup:
        setup_api_key()
        return

    if not args.url:
        parser.print_help()
        sys.exit(1)

    url: str = args.url

    # ── YouTube path ─────────────────────────────────────────────────────
    if is_youtube_url(url):
        require_api_key()
        print("\nDetected YouTube URL\n")
        yt = download_youtube_audio(url)

        print("Starting transcription...")
        print("This may take several minutes depending on video length.\n")

        try:
            t0 = time.time()
            segments, duration = transcribe(yt["audio_path"], is_file=True)
            elapsed = int(time.time() - t0)
        finally:
            # Clean up temp file
            try:
                os.unlink(yt["audio_path"])
            except OSError:
                pass

        print(f"\nTranscription complete in {elapsed} seconds.")
        print(f"Duration: {format_timestamp(duration)}")
        print(f"Segments: {len(segments)}")

        md = generate_markdown(yt["title"], yt["channel"], segments, duration)
        filename = args.output or f"{sanitize_filename(yt['title'])}.md"
        Path(filename).write_text(md, encoding="utf-8")
        print(f"\nSaved to: {filename}")
        return

    # ── Resolve Apple Podcasts URLs to RSS ──────────────────────────────
    apple_episode_id = None
    original_url = url
    if parse_apple_podcasts_url(url):
        print("\nDetected Apple Podcasts URL\n")
        url, apple_episode_id = resolve_apple_podcasts_url(url)

    # ── Podcast RSS path ─────────────────────────────────────────────────
    print(f"Fetching podcast feed: {url}\n")

    podcast_name, episodes = parse_feed(url)
    print(f"Podcast: {podcast_name}")
    print(f"Found {len(episodes)} episodes\n")

    if not episodes:
        print("No episodes found in feed.", file=sys.stderr)
        sys.exit(1)

    # Filter by search query
    if args.search:
        query = args.search.lower()
        episodes = [
            ep for ep in episodes
            if query in ep.title.lower() or query in ep.description.lower()
        ]
        print(f'Found {len(episodes)} episodes matching "{args.search}":\n')
        if not episodes:
            print("No episodes match your search.", file=sys.stderr)
            sys.exit(1)

    # ── Apple episode: resolve before listing ────────────────────────────
    # When the user shared a specific episode link, try to match it now
    # so we can skip the episode listing entirely.
    apple_selected = None
    from_scraped_url = False
    if apple_episode_id:
        apple_info = scrape_apple_episode_info(original_url)
        if apple_info and apple_info.get("title"):
            ep_title = apple_info["title"]
            print(f"Looking for episode: {ep_title}")

            # Exact match (case-insensitive, stripped)
            for ep in episodes:
                if ep.title.strip().lower() == ep_title.strip().lower():
                    apple_selected = ep
                    break

            # Fuzzy match: normalize and try substring containment
            if apple_selected is None:
                norm_apple = normalize_for_matching(ep_title)
                for ep in episodes:
                    norm_ep = normalize_for_matching(ep.title)
                    if norm_apple == norm_ep or norm_apple in norm_ep or norm_ep in norm_apple:
                        print(f'\nFuzzy match found: "{ep.title}"')
                        try:
                            confirm = input("Is this the correct episode? (y/n): ").strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            confirm = "n"
                        if confirm == "y":
                            apple_selected = ep
                        break

        if apple_selected is None and apple_info and apple_info.get("audio_url"):
            # Episode not in RSS feed (too old), but we have the audio URL from Apple
            title = apple_info.get("title", "Unknown Episode")
            print(f"\nEpisode not in RSS feed, using scraped audio URL from Apple Podcasts page.")
            print(f"Title: {title}")
            apple_selected = Episode(
                title=title,
                audio_url=apple_info["audio_url"],
                publish_date="",
                description="",
                duration="",
            )
            from_scraped_url = True

        if apple_selected is None:
            # Don't silently fall back to the wrong episode
            print(f"\nError: Could not find the specific episode from this Apple Podcasts link.", file=sys.stderr)
            print(f"Suggestions:", file=sys.stderr)
            print(f'  - Try: podscript "{url}" --search "keyword from episode title"', file=sys.stderr)
            print(f'  - Try: podscript "{url}" --list   (to browse episodes)', file=sys.stderr)
            sys.exit(1)

    # ── Display episode list (skip when a specific Apple episode is already selected) ──
    if apple_selected is None:
        display_limit = len(episodes) if args.search else 30
        for i, ep in enumerate(episodes[:display_limit]):
            try:
                dt = datetime.strptime(ep.publish_date[:16], "%a, %d %b %Y")
                date_str = f"{dt.month}/{dt.day}/{dt.year}"
            except (ValueError, IndexError):
                date_str = ep.publish_date[:20] if ep.publish_date else "Unknown date"
            dur = ep.duration or "Unknown duration"
            print(f"  {i + 1:3}. {ep.title}")
            print(f"       {date_str} | {dur}")

        if len(episodes) > display_limit:
            print(f"\n  ... and {len(episodes) - display_limit} more episodes (use --search to filter)")

    # If --list or (--search without --episode), stop here
    if args.list_episodes or (args.search and args.episode is None and not apple_episode_id):
        return

    require_api_key()

    # Select episode
    if apple_selected is not None:
        selected = apple_selected
    elif args.episode is not None:
        if args.episode < 1 or args.episode > len(episodes):
            print(f"\nInvalid episode number. Choose between 1 and {len(episodes)}.", file=sys.stderr)
            sys.exit(1)
        selected = episodes[args.episode - 1]
    else:
        # Default: latest
        selected = episodes[0]
        print("\nUsing most recent episode.")

    print(f"\nSelected: {selected.title}")
    print(f"Audio URL: {selected.audio_url}\n")
    print("Starting transcription...")
    print("This may take several minutes depending on episode length.\n")

    # If the audio URL came from scraping Apple (not RSS), download to temp file
    # to avoid URL expiry / auth issues with ElevenLabs cloud_storage_url.
    if from_scraped_url:
        print("Downloading audio from Apple Podcasts...")
        temp_audio = os.path.join(tempfile.gettempdir(), f"podscript-apple-{os.getpid()}.mp3")
        try:
            dl_resp = requests.get(selected.audio_url, timeout=600, stream=True)
            dl_resp.raise_for_status()
            with open(temp_audio, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(temp_audio) / (1024 * 1024)
            print(f"Downloaded: {size_mb:.1f} MB\n")
        except Exception as e:
            print(f"Error downloading audio: {e}", file=sys.stderr)
            sys.exit(1)
        try:
            t0 = time.time()
            segments, duration = transcribe(temp_audio, is_file=True)
            elapsed = int(time.time() - t0)
        finally:
            try:
                os.unlink(temp_audio)
            except OSError:
                pass
    else:
        t0 = time.time()
        segments, duration = transcribe(selected.audio_url)
        elapsed = int(time.time() - t0)

    print(f"\nTranscription complete in {elapsed} seconds.")
    print(f"Duration: {format_timestamp(duration)}")
    print(f"Segments: {len(segments)}")

    md = generate_markdown(selected.title, podcast_name, segments, duration)
    filename = args.output or f"{sanitize_filename(selected.title)}.md"
    Path(filename).write_text(md, encoding="utf-8")
    print(f"\nSaved to: {filename}")


if __name__ == "__main__":
    main()
