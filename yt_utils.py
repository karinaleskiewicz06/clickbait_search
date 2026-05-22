import re
import requests
from urllib.parse import parse_qs, urlparse


def get_video_id(url):
    """Get video ID from a YouTube URL."""
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]
    return None


def get_title(video_id):
    """Read title from the watch page HTML."""
    r = requests.get(
        f"https://www.youtube.com/watch?v={video_id}",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    r.raise_for_status()
    m = re.search(r'<meta property="og:title" content="([^"]+)"', r.text)
    if m:
        return m.group(1)
    m = re.search(r"<title>([^<]+)</title>", r.text)
    if m:
        return m.group(1).replace(" - YouTube", "").strip()
    return None
