import re
import sys
import requests
from urllib.parse import parse_qs, urlparse

from clickbait_terms import PHRASES, WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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


def find_buzzwords(title):
    """Find clickbait phrases and words in the title."""
    t = title.lower()
    hits = []
    for p in PHRASES:
        if p in t:
            hits.append(p)
    for w in WORDS:
        if re.search(r"\b" + re.escape(w) + r"\b", t):
            hits.append(w)
    return hits


def analyze_title(title):
    """Caps, punctuation, VADER, and clickbait word checks."""
    letters = [c for c in title if c.isalpha()]
    caps_pct = 0
    if letters:
        caps_pct = round(100 * sum(c.isupper() for c in letters) / len(letters), 1)

    excl = title.count("!")
    quest = title.count("?")
    vader = SentimentIntensityAnalyzer().polarity_scores(title)["compound"]
    buzz = find_buzzwords(title)

    # prosty score stylu tytulu (im wyzszy tym bardziej clickbaitowy)
    # analizuje czy tytul jest wielkimi literami, czy ma !, ? oraz znaczeniowo(biblio vader)
    score = 0
    score += min(len(buzz) * 0.18, 0.45)
    if caps_pct >= 40:
        score += 0.2
    elif caps_pct >= 25:
        score += 0.1
    if excl >= 2:
        score += 0.15
    elif excl >= 1:
        score += 0.08
    if quest >= 2:
        score += 0.08
    if abs(vader) >= 0.5:
        score += 0.1
    score = round(min(score, 1), 2)

    if score < 0.25:
        label = "neutral"
    elif score < 0.5:
        label = "mild"
    else:
        label = "strong"

    print("\n" + "=" * 50)
    print("TITLE ANALYSIS")
    print("=" * 50)
    print("Title:", title)
    print("Caps %:", caps_pct)
    print("Exclamation marks:", excl)
    print("Question marks:", quest)
    print("VADER compound:", round(vader, 3))
    print("Clickbait terms:", ", ".join(buzz) if buzz else "none")
    print("Style score:", score, label)
    print("=" * 50)


url = sys.argv[1] if len(sys.argv) > 1 else input("YouTube URL: ")
url = url.strip()

video_id = get_video_id(url)
if not video_id:
    print("Could not parse video ID from URL.")
    sys.exit(1)

title = get_title(video_id)
if not title:
    print("Could not fetch video title.")
    sys.exit(1)

# analiza stylu tytulu
analyze_title(title)

#transkrypcja vs tytul 

# model do embeddingow tekstu
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])

# dzielimy transkrypcje na bloki ~45 sekund
chunks = []
text = ""
start = None

for snippet in transcript.snippets:
    if start is None:
        start = snippet.start
        text = snippet.text + " "
    elif snippet.start - start <= 45:
        text += snippet.text + " "
    else:
        chunks.append({"text": text.strip(), "start": start})
        start = snippet.start
        text = snippet.text + " "

if text:
    chunks.append({"text": text.strip(), "start": start})

print(f"\nSplit into {len(chunks)} chunks.")

# embeddingi chunkow i tytulu, potem cosine similarity
texts = [c["text"] for c in chunks]
emb_chunks = model.encode(texts)
emb_title = model.encode([title])
scores = cosine_similarity(emb_title, emb_chunks)[0]

best_i = int(np.argmax(scores))
best_chunk = chunks[best_i]
max_pct = round(scores[best_i] * 100, 1)

total_time = chunks[-1]["start"] + 45.0
time_pct = round((best_chunk["start"] / total_time) * 100, 1)

threshold = 0.30
good_blocks = sum(1 for s in scores if s >= threshold)
density_pct = round((good_blocks / len(chunks)) * 100, 1)
std_pct = round(float(np.std(scores)) * 100, 1)

print("\n" + "=" * 50)
print("TRANSCRIPT ANALYSIS")
print("=" * 50)
print("Max similarity to title:", max_pct, "%")
print("Best match at", time_pct, "% of video length")
print("Topic density:", density_pct, "% of chunks above threshold")
print("Score spread (std):", std_pct)
print("\nBest chunk starts at", best_chunk["start"], "s")
print(">", best_chunk["text"])
print("=" * 50)
