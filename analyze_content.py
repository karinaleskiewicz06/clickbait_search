import re
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from youtube_transcript_api import YouTubeTranscriptApi

from yt_utils import get_video_id, get_title


# Cosine similarity thresholds tuned for sentence-transformers/all-MiniLM-L6-v2.
# Typical similarity between a YouTube title and a transcript chunk:
#   < 0.20  unrelated
#   0.20 - 0.30  loosely related (same general field)
#   0.30 - 0.45  clearly related (talking around the topic)
#   >= 0.45  talking directly about the topic
WEAK_THRESHOLD = 0.25       # topic is at least loosely brought up
STRONG_THRESHOLD = 0.40     # topic is clearly being discussed

CHUNK_SECONDS = 45          # length of one transcript block
TEXT_PREVIEW_LEN = 200      # how many chars of a chunk to show as a preview


def format_time(seconds):
    """Convert seconds (float) to HH:MM:SS string."""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def preview(text, limit=TEXT_PREVIEW_LEN):
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def fetch_chunks(video_id, chunk_seconds=CHUNK_SECONDS, languages=("en",)):
    """Pobiera transkrypcje i dzieli ja na bloki ~chunk_seconds sekund."""
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))

    chunks = []
    text = ""
    start = None

    for snippet in transcript.snippets:
        # usuwa adnotacje typu [Music], (applause) i symbole nutek
        clean_text = re.sub(r"\[.*?\]|\(.*?\)|♪", "", snippet.text)
        clean_text = " ".join(clean_text.split())

        if not clean_text:
            continue

        if start is None:
            start = snippet.start
            text = clean_text + " "
        elif snippet.start - start <= chunk_seconds:
            text += clean_text + " "
        else:
            chunks.append({"text": text.strip(), "start": start})
            start = snippet.start
            text = clean_text + " "

    if text.strip():
        chunks.append({"text": text.strip(), "start": start})

    return chunks


def find_topic_segments(chunks, scores, threshold):
    segments = []
    i = 0
    n = len(chunks)
    while i < n:
        if scores[i] >= threshold:
            j = i
            while j + 1 < n and scores[j + 1] >= threshold:
                j += 1

            seg_start = chunks[i]["start"]
            if j + 1 < n:
                seg_end = chunks[j + 1]["start"]
            else:
                seg_end = chunks[j]["start"] + CHUNK_SECONDS

            peak_offset = int(np.argmax(scores[i:j + 1]))
            peak_idx = i + peak_offset
            segments.append({
                "start": seg_start,
                "end": seg_end,
                "peak_score": float(scores[peak_idx]),
                "peak_idx": peak_idx,
            })
            i = j + 1
        else:
            i += 1
    return segments


def first_index_above(scores, threshold):
    for i, s in enumerate(scores):
        if s >= threshold:
            return i
    return None


def analyze_content(title, video_id):
    """Porownuje tytul z trescia transkrypcji przez embeddingi i cosine similarity."""
    print("\nFetching transcript...")
    try:
        chunks = fetch_chunks(video_id)
    except Exception as e:
        print("Could not fetch transcript for this video.")
        print("Reason:", e)
        print("(The video may not have captions, or they are in another language.)")
        return

    if not chunks:
        print("Transcript is empty, skipping content analysis.")
        return

    print(f"Split into {len(chunks)} chunks.")
    print("Loading sentence-transformer model (first run downloads ~90 MB)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Computing embeddings and similarities...")
    texts = [c["text"] for c in chunks]
    emb_chunks = model.encode(texts)
    emb_title = model.encode([title])
    scores = cosine_similarity(emb_title, emb_chunks)[0]

    total_time = chunks[-1]["start"] + CHUNK_SECONDS

    best_i = int(np.argmax(scores))
    best_chunk = chunks[best_i]
    best_pct = round(float(scores[best_i]) * 100, 1)

    first_weak_i = first_index_above(scores, WEAK_THRESHOLD)
    first_strong_i = first_index_above(scores, STRONG_THRESHOLD)

    weak_count = int(np.sum(scores >= WEAK_THRESHOLD))
    strong_count = int(np.sum(scores >= STRONG_THRESHOLD))
    weak_density = round(weak_count / len(chunks) * 100, 1)
    strong_density = round(strong_count / len(chunks) * 100, 1)
    std_pct = round(float(np.std(scores)) * 100, 1)

    print("\n" + "=" * 60)
    print("TRANSCRIPT ANALYSIS")
    print("=" * 60)
    print("Title:", title)
    print("Video length (approx):", format_time(total_time))
    print(f"Thresholds: weak >= {WEAK_THRESHOLD}, strong >= {STRONG_THRESHOLD}")

    print("\nBEST MATCH")
    print(f"  Time:       {format_time(best_chunk['start'])}")
    print(f"  Similarity: {best_pct} %")
    print(f"  Excerpt:    {preview(best_chunk['text'])}")

    print("\nFIRST APPEARANCE OF THE TOPIC")
    if first_weak_i is not None:
        c = chunks[first_weak_i]
        s_pct = round(float(scores[first_weak_i]) * 100, 1)
        print(f"  First loose mention (>= {WEAK_THRESHOLD}): "
              f"{format_time(c['start'])}  ({s_pct} %)")
        print(f"    > {preview(c['text'])}")
    else:
        print(f"  First loose mention (>= {WEAK_THRESHOLD}): not found")

    if first_strong_i is not None:
        c = chunks[first_strong_i]
        s_pct = round(float(scores[first_strong_i]) * 100, 1)
        first_strong_time_pct = round(c["start"] / total_time * 100, 1)
        print(f"  First on-topic (>= {STRONG_THRESHOLD}):    "
              f"{format_time(c['start'])}  ({s_pct} %)  "
              f"[{first_strong_time_pct} % into the video]")
        print(f"    > {preview(c['text'])}")
    else:
        print(f"  First on-topic (>= {STRONG_THRESHOLD}):    not found "
              "(title topic never clearly appears -> possible clickbait)")

    print("\nTOPIC COVERAGE")
    print(f"  Strong (>= {STRONG_THRESHOLD}): {strong_count}/{len(chunks)} chunks "
          f"({strong_density} %)")
    print(f"  Weak   (>= {WEAK_THRESHOLD}): {weak_count}/{len(chunks)} chunks "
          f"({weak_density} %)")
    print(f"  Score spread (std x 100): {std_pct}")

    print("\nON-TOPIC MOMENTS THROUGHOUT THE VIDEO")
    strong_segments = find_topic_segments(chunks, scores, STRONG_THRESHOLD)
    if strong_segments:
        print(f"  Strong segments (>= {STRONG_THRESHOLD}):")
        for seg in strong_segments:
            peak_pct = round(seg["peak_score"] * 100, 1)
            print(f"    {format_time(seg['start'])} - {format_time(seg['end'])}"
                  f"   peak {peak_pct} %")
    else:
        print(f"  No strong segments (>= {STRONG_THRESHOLD}).")

    weak_segments = find_topic_segments(chunks, scores, WEAK_THRESHOLD)
    if weak_segments:
        print(f"  Weak segments (>= {WEAK_THRESHOLD}):")
        for seg in weak_segments:
            peak_pct = round(seg["peak_score"] * 100, 1)
            print(f"    {format_time(seg['start'])} - {format_time(seg['end'])}"
                  f"   peak {peak_pct} %")
    else:
        print(f"  No weak segments (>= {WEAK_THRESHOLD}) either.")

    print("=" * 60)


def main():
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

    analyze_content(title, video_id)


if __name__ == "__main__":
    main()
