import re
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
from clickbait_terms import PHRASES, WORDS
from sklearn.metrics.pairwise import cosine_similarity
from youtube_transcript_api import YouTubeTranscriptApi

from yt_utils import get_video_id, get_title


# Cosine similarity thresholds tuned for sentence-transformers/all-mpnet-base-v2
# with 0.7 embedding + 0.3 lexical score blending.
# Typical similarity between a YouTube title and a transcript chunk:
#   < 0.25  unrelated
#   0.25 - 0.35  loosely related (same general field)
#   0.35 - 0.50  clearly related (talking around the topic)
#   >= 0.50  talking directly about the topic
WEAK_THRESHOLD = 0.30
STRONG_THRESHOLD = 0.45

# Final "worth watching?" verdict thresholds (used with analyze_title score).
TITLE_CLICKBAIT_THRESHOLD = 0.50
TOPIC_DENSITY_MIN = 20.0
FIRST_STRONG_LATE_PCT = 65.0
AVG_MATCH_MIN = 25.0
STRONG_DENSITY_FALLBACK = 15.0

VERDICT_NO = "no"
VERDICT_PROBABLY_NOT = "probably_not"
VERDICT_YES_BUT = "yes_but"
VERDICT_WORTH_IT = "worth_it"

CHUNK_SECONDS = 45
FINE_STRIDE_SECONDS = 15
TEXT_PREVIEW_LEN = 200
EMBED_WEIGHT = 0.7
LEXICAL_WEIGHT = 0.3
SMOOTH_WINDOW = 3

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

NOISE_BRACKET = re.compile(
    r"\[(?:Music|Singing|Applause|Laughter|Laughing|Crowd|Cheering|Silence)\]",
    re.IGNORECASE,
)
NOISE_PAREN = re.compile(
    r"\((?:applause|laughter|laughing|music|inaudible|crosstalk)\)",
    re.IGNORECASE,
)

STOPWORDS = frozenset({
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "is",
    "it", "this", "that", "how", "why", "what", "when", "with", "from", "by",
    "vs", "are", "was", "be", "your", "you", "my", "we", "our",
})


@dataclass
class ContentAnalysis:
    chunks: list
    scores: list
    raw_scores: list
    best_index: int
    best_chunk: dict
    peak_match_pct: int
    avg_match_pct: int
    topic_density_pct: int
    signal_chaos_pct: int
    first_weak_index: Optional[int]
    first_strong_index: Optional[int]
    first_strong_pct: float
    total_time: float
    weak_segments: list
    strong_segments: list
    title: str
    video_id: str

    def to_dict(self):
        data = asdict(self)
        # Backward compatibility for callers expecting the old metric name.
        data["peak_alignment_pct"] = self.peak_match_pct
        return data


@lru_cache(maxsize=1)
def get_embedding_model():
    # Lazy import: avoids loading transformers (and its optional vision deps)
    # when Streamlit imports this module on startup.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def format_time(seconds):
    """Convert seconds (float) to HH:MM:SS string."""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def preview(text, limit=TEXT_PREVIEW_LEN):
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def clean_snippet_text(text):
    text = NOISE_BRACKET.sub("", text)
    text = NOISE_PAREN.sub("", text)
    text = text.replace("♪", "")
    return " ".join(text.split())


def clean_title_for_embedding(title):
    cleaned = title.lower()
    for phrase in sorted(PHRASES, key=len, reverse=True):
        cleaned = cleaned.replace(phrase, " ")
    for word in WORDS:
        cleaned = re.sub(r"\b" + re.escape(word) + r"\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -|:,.")
    return cleaned if len(cleaned) >= 3 else title


def extract_lexical_terms(title, cleaned_title):
    terms = set()
    for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', title):
        terms.add(match.group(1) or match.group(2))
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", title):
        terms.add(match.group(0))
    for match in re.finditer(r"\b[A-Z]{2,}\b", title):
        terms.add(match.group(0))
    for match in re.finditer(r"\b\d[\w.]*\b", title):
        terms.add(match.group(0))
    for word in cleaned_title.split():
        if len(word) > 3 and word not in STOPWORDS:
            terms.add(word)
    return [term for term in terms if len(term) >= 2]


def lexical_score(text, terms):
    if not terms:
        return 0.0
    lowered = text.lower()
    hits = 0
    for term in terms:
        if " " in term:
            if term.lower() in lowered:
                hits += 1
        elif re.search(r"\b" + re.escape(term.lower()) + r"\b", lowered):
            hits += 1
    return hits / len(terms)


def fetch_transcript_snippets(video_id, languages=("en",)):
    """Fetch transcript snippets with cleaned text, start, and duration."""
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    snippets = []
    for snippet in transcript.snippets:
        clean_text = clean_snippet_text(snippet.text)
        if not clean_text:
            continue
        snippets.append({
            "text": clean_text,
            "start": snippet.start,
            "duration": snippet.duration,
        })
    if not snippets:
        return [], 0.0
    total_time = max(s["start"] + s["duration"] for s in snippets)
    return snippets, total_time


def build_time_window_chunks(snippets, window_sec, stride_sec):
    """Build overlapping transcript chunks aligned to fixed time windows."""
    if not snippets:
        return []

    end_time = max(s["start"] + s["duration"] for s in snippets)
    start_time = snippets[0]["start"]
    chunks = []
    window_start = start_time

    while window_start < end_time:
        window_end = min(window_start + window_sec, end_time)
        parts = []
        for snippet in snippets:
            snippet_end = snippet["start"] + snippet["duration"]
            if snippet["start"] < window_end and snippet_end > window_start:
                parts.append(snippet["text"])

        if parts:
            chunks.append({
                "text": " ".join(parts),
                "start": window_start,
                "end": window_end,
            })
        window_start += stride_sec

    return chunks


def pool_scores_to_primary(fine_chunks, fine_scores, primary_chunks):
    """Max-pool fine-grained scores onto primary chunk intervals."""
    pooled = []
    for primary in primary_chunks:
        primary_end = primary["end"]
        best = 0.0
        for fine, score in zip(fine_chunks, fine_scores):
            if fine["start"] < primary_end and fine["end"] > primary["start"]:
                best = max(best, score)
        pooled.append(best)
    return pooled


def smooth_scores(scores, window=SMOOTH_WINDOW):
    if len(scores) <= 1:
        return list(scores)
    arr = np.array(scores, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same").tolist()


def compute_chunk_scores(model, title, cleaned_title, lexical_terms, chunks):
    texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = model.encode(texts)
    title_embeddings = model.encode([title, cleaned_title])
    full_similarity = cosine_similarity(title_embeddings[:1], chunk_embeddings)[0]
    clean_similarity = cosine_similarity(title_embeddings[1:], chunk_embeddings)[0]
    embed_scores = np.maximum(full_similarity, clean_similarity)

    combined = []
    for embed_score, chunk in zip(embed_scores, chunks):
        lex_score = lexical_score(chunk["text"], lexical_terms)
        combined.append(
            EMBED_WEIGHT * float(embed_score) + LEXICAL_WEIGHT * float(lex_score)
        )
    return combined


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
                seg_end = chunks[j].get("end", chunks[j]["start"] + CHUNK_SECONDS)

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
    for index, score in enumerate(scores):
        if score >= threshold:
            return index
    return None


def compute_verdict_signals(title_score, content):
    """Derive boolean signals used by the final watch recommendation."""
    scores = content["scores"]
    peak = scores[content["best_index"]]

    title_clickbait = title_score >= TITLE_CLICKBAIT_THRESHOLD
    delivers_weak = (
        peak >= WEAK_THRESHOLD or content["first_weak_index"] is not None
    )
    delivers_strong = (
        peak >= STRONG_THRESHOLD
        or (
            content["first_strong_index"] is not None
            and content["topic_density_pct"] >= STRONG_DENSITY_FALLBACK
        )
    )
    density_ok = content["topic_density_pct"] >= TOPIC_DENSITY_MIN
    topic_late = (
        content["first_strong_index"] is not None
        and content["first_strong_pct"] > FIRST_STRONG_LATE_PCT
    )
    avg_ok = content["avg_match_pct"] >= AVG_MATCH_MIN
    good_structure = density_ok and not topic_late and avg_ok

    return {
        "title_clickbait": title_clickbait,
        "peak_score": float(peak),
        "delivers_weak": delivers_weak,
        "delivers_strong": delivers_strong,
        "density_ok": density_ok,
        "topic_late": topic_late,
        "avg_ok": avg_ok,
        "good_structure": good_structure,
    }


def compute_watch_verdict(title_score, content):
    """
    Return the final recommendation from title + content analysis.

    Decision order:
      1. No weak topic signal anywhere -> NO
      2. Weak but no strong delivery -> PROBABLY NOT
      3. Strong but bad structure -> YES, BUT
      4. Strong + good structure + clickbait title -> YES, BUT
      5. Strong + good structure + honest title -> WORTH IT
    """
    signals = compute_verdict_signals(title_score, content)

    if not signals["delivers_weak"]:
        verdict = VERDICT_NO
        if signals["title_clickbait"]:
            headline = "### ❌ NO."
            message = (
                "This video is pure clickbait. The headline was manufactured just to get your view, "
                "but the actual content completely fails to deliver on its promise."
            )
        else:
            headline = "### ❌ NO."
            message = (
                "The title layout seems normal, but it's a completely wrong headline. "
                "The video covers an entirely different topic."
            )
        reason = "no_topic"
    elif not signals["delivers_strong"]:
        verdict = VERDICT_PROBABLY_NOT
        headline = "### 🧐 PROBABLY NOT."
        if signals["title_clickbait"]:
            message = (
                "The headline is heavily exaggerated. The creator only briefly and loosely touches "
                "upon the topic, without offering any strong or real substance."
            )
        else:
            message = (
                f"The title is honest, but the video is poorly focused "
                f"and has very low topic density ({content['topic_density_pct']}%)."
            )
        reason = "weak_delivery"
    elif not signals["good_structure"]:
        verdict = VERDICT_YES_BUT
        headline = "### ⚠️ YES, BUT..."
        parts = []
        if not signals["density_ok"]:
            parts.append(f"focus density is too low ({content['topic_density_pct']}%)")
        if signals["topic_late"]:
            parts.append(f"the main point is buried too deep ({int(content['first_strong_pct'])}%)")
        if not signals["avg_ok"]:
            parts.append(f"overall relevance stays low ({content['avg_match_pct']}%)")
        detail = ", ".join(parts) if parts else "the topic is not sustained through the video"
        message = f"The video touches the topic, but it is badly structured: {detail}."
        reason = "bad_structure"
    elif signals["title_clickbait"]:
        verdict = VERDICT_YES_BUT
        headline = "### ⚠️ YES, BUT..."
        message = (
            "The headline is heavily exaggerated and sensationalized to generate hype. "
            "However, the video does actually discuss the promised topic in comprehensive detail."
        )
        reason = "clickbait_title"
    else:
        verdict = VERDICT_WORTH_IT
        headline = "### ✅ DEFINITELY WORTH IT!"
        message = (
            "This is a perfectly genuine video. The title is honest, accurate, "
            "and backed up by solid, relevant content."
        )
        reason = "genuine"

    return {
        "verdict": verdict,
        "headline": headline,
        "message": message,
        "reason": reason,
        "signals": signals,
    }


def compute_content_analysis(title, video_id, languages=("en",)):
    """Compare title with transcript content via embeddings and lexical matching."""
    try:
        snippets, total_time = fetch_transcript_snippets(video_id, languages=languages)
    except Exception:
        return None

    if not snippets:
        return None

    primary_chunks = build_time_window_chunks(
        snippets, window_sec=CHUNK_SECONDS, stride_sec=CHUNK_SECONDS
    )
    fine_chunks = build_time_window_chunks(
        snippets, window_sec=CHUNK_SECONDS, stride_sec=FINE_STRIDE_SECONDS
    )
    if not primary_chunks:
        return None

    cleaned_title = clean_title_for_embedding(title)
    lexical_terms = extract_lexical_terms(title, cleaned_title)
    model = get_embedding_model()

    fine_raw_scores = compute_chunk_scores(
        model, title, cleaned_title, lexical_terms, fine_chunks
    )
    raw_scores = pool_scores_to_primary(fine_chunks, fine_raw_scores, primary_chunks)
    scores = smooth_scores(raw_scores)

    best_index = int(np.argmax(scores))
    best_chunk = primary_chunks[best_index]
    best_pct = round(float(scores[best_index]) * 100, 1)

    first_weak_index = first_index_above(scores, WEAK_THRESHOLD)
    first_strong_index = first_index_above(scores, STRONG_THRESHOLD)

    weak_count = int(np.sum(np.array(scores) >= WEAK_THRESHOLD))
    strong_count = int(np.sum(np.array(scores) >= STRONG_THRESHOLD))
    weak_density = round(weak_count / len(primary_chunks) * 100, 1)
    strong_density = round(strong_count / len(primary_chunks) * 100, 1)
    std_pct = round(float(np.std(scores)) * 100, 1)

    if first_strong_index is not None:
        first_strong_pct = round(
            primary_chunks[first_strong_index]["start"] / total_time * 100, 1
        )
    else:
        first_strong_pct = 100.0

    return ContentAnalysis(
        chunks=primary_chunks,
        scores=[float(score) for score in scores],
        raw_scores=[float(score) for score in raw_scores],
        best_index=best_index,
        best_chunk=best_chunk,
        peak_match_pct=int(best_pct),
        avg_match_pct=int(round(float(np.mean(scores)) * 100)),
        topic_density_pct=int(strong_density),
        signal_chaos_pct=int(std_pct),
        first_weak_index=first_weak_index,
        first_strong_index=first_strong_index,
        first_strong_pct=first_strong_pct,
        total_time=total_time,
        weak_segments=find_topic_segments(primary_chunks, scores, WEAK_THRESHOLD),
        strong_segments=find_topic_segments(primary_chunks, scores, STRONG_THRESHOLD),
        title=title,
        video_id=video_id,
    )


def print_content_report(result: ContentAnalysis):
    chunks = result.chunks
    scores = result.scores
    best_chunk = result.best_chunk
    best_pct = result.peak_match_pct

    print("\n" + "=" * 60)
    print("TRANSCRIPT ANALYSIS")
    print("=" * 60)
    print("Title:", result.title)
    print("Video length:", format_time(result.total_time))
    print(f"Model: {MODEL_NAME}")
    print(f"Thresholds: weak >= {WEAK_THRESHOLD}, strong >= {STRONG_THRESHOLD}")
    print(f"Chunks: {len(chunks)} primary ({CHUNK_SECONDS}s), "
          f"scored with {FINE_STRIDE_SECONDS}s stride overlap")

    print("\nBEST MATCH")
    print(f"  Time:       {format_time(best_chunk['start'])}")
    print(f"  Similarity: {best_pct} %")
    print(f"  Excerpt:    {preview(best_chunk['text'])}")

    print("\nFIRST APPEARANCE OF THE TOPIC")
    if result.first_weak_index is not None:
        chunk = chunks[result.first_weak_index]
        score_pct = round(float(scores[result.first_weak_index]) * 100, 1)
        print(f"  First loose mention (>= {WEAK_THRESHOLD}): "
              f"{format_time(chunk['start'])}  ({score_pct} %)")
        print(f"    > {preview(chunk['text'])}")
    else:
        print(f"  First loose mention (>= {WEAK_THRESHOLD}): not found")

    if result.first_strong_index is not None:
        chunk = chunks[result.first_strong_index]
        score_pct = round(float(scores[result.first_strong_index]) * 100, 1)
        print(f"  First on-topic (>= {STRONG_THRESHOLD}):    "
              f"{format_time(chunk['start'])}  ({score_pct} %)  "
              f"[{result.first_strong_pct} % into the video]")
        print(f"    > {preview(chunk['text'])}")
    else:
        print(f"  First on-topic (>= {STRONG_THRESHOLD}):    not found "
              "(title topic never clearly appears -> possible clickbait)")

    weak_count = int(np.sum(np.array(scores) >= WEAK_THRESHOLD))
    strong_count = int(np.sum(np.array(scores) >= STRONG_THRESHOLD))
    weak_density = round(weak_count / len(chunks) * 100, 1)
    strong_density = round(strong_count / len(chunks) * 100, 1)

    print("\nTOPIC COVERAGE")
    print(f"  Strong (>= {STRONG_THRESHOLD}): {strong_count}/{len(chunks)} chunks "
          f"({strong_density} %)")
    print(f"  Weak   (>= {WEAK_THRESHOLD}): {weak_count}/{len(chunks)} chunks "
          f"({weak_density} %)")
    print(f"  Score spread (std x 100): {result.signal_chaos_pct}")

    print("\nON-TOPIC MOMENTS THROUGHOUT THE VIDEO")
    if result.strong_segments:
        print(f"  Strong segments (>= {STRONG_THRESHOLD}):")
        for segment in result.strong_segments:
            peak_pct = round(segment["peak_score"] * 100, 1)
            print(f"    {format_time(segment['start'])} - {format_time(segment['end'])}"
                  f"   peak {peak_pct} %")
    else:
        print(f"  No strong segments (>= {STRONG_THRESHOLD}).")

    if result.weak_segments:
        print(f"  Weak segments (>= {WEAK_THRESHOLD}):")
        for segment in result.weak_segments:
            peak_pct = round(segment["peak_score"] * 100, 1)
            print(f"    {format_time(segment['start'])} - {format_time(segment['end'])}"
                  f"   peak {peak_pct} %")
    else:
        print(f"  No weak segments (>= {WEAK_THRESHOLD}) either.")

    print("=" * 60)


def analyze_content(title, video_id, verbose=True):
    """Run content analysis and optionally print a CLI report."""
    if verbose:
        print("\nFetching transcript...")
        print(f"Loading sentence-transformer model (first run downloads ~420 MB)...")

    result = compute_content_analysis(title, video_id)
    if result is None:
        if verbose:
            print("Could not fetch transcript for this video.")
            print("(The video may not have captions, or they are in another language.)")
        return None

    if verbose:
        print(f"Split into {len(result.chunks)} primary chunks.")
        print("Computing embeddings and similarities...")
        print_content_report(result)

    return result.to_dict()


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
