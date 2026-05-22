'''
app.py — main entry point with menu 
analyze_title.py — standalone title analyzer 
analyze_content.py — standalone content analyzer 
yt_utils.py — shared helpers 
clickbait_terms.py — list of clickbait words/phrases
'''
import sys
from yt_utils import get_video_id, get_title
from analyze_title import analyze_title
from analyze_content import analyze_content


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
    analyze_content(title, video_id)


if __name__ == "__main__":
    main()
