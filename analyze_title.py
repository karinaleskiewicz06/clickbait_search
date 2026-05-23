import re
import sys

from clickbait_terms import PHRASES, WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from yt_utils import get_video_id, get_title


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


def caps_percentage(title):
    """Procent wielkich liter w tytule, ignorujac pierwsza litere kazdego slowa.

    Dzieki temu zwykly title case ("How To Stop AI") nie podbija wyniku,
    a krzyczace tytuly typu "THIS IS AMAZING" nadal sie wykrywaja.
    """
    caps_count = 0
    total_count = 0
    for word in title.split():
        for c in word[1:]:
            if c.isalpha():
                total_count += 1
                if c.isupper():
                    caps_count += 1
    if not total_count:
        return 0.0
    return round(100 * caps_count / total_count, 1)


def analyze_title(title):
    """Caps, punctuation, VADER, and clickbait word checks."""
    caps_pct = caps_percentage(title)

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

    return {
        "caps_pct": caps_pct,
        "exclamation_marks": excl,
        "question_marks": quest,
        "vader_score": vader,
        "buzzwords": buzz,
        "score": score,
        "label": label
    }


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

    analyze_title(title)


if __name__ == "__main__":
    main()
