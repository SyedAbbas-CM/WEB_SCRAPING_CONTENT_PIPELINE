#!/usr/bin/env python3
"""
Polite scraper for old.reddit JSON endpoints (no proxies).

Examples:
  python scripts/old_reddit_json_scraper.py --subreddit python --pages 1 --limit 5 --out out
"""

import argparse
import json
import pathlib
import random
import time
from typing import Iterator, Optional

import requests


BASE = "https://old.reddit.com"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def sleep_jitter(a: float = 1.0, b: float = 3.0) -> None:
    time.sleep(random.uniform(a, b))


def get_json(session: requests.Session, url: str, max_retries: int = 5) -> dict:
    for i in range(max_retries):
        r = session.get(url, headers=DEFAULT_HEADERS, timeout=30)
        text = r.text or ""
        if r.status_code == 200 and "Just a moment" not in text:
            return r.json()
        if r.status_code in (403, 429) or "Just a moment" in text:
            # exponential backoff with jitter
            time.sleep((2 ** i) * 2 + random.uniform(0, 2))
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after retries: {url}")


def iter_sub_posts(
    session: requests.Session,
    subreddit: str,
    sort: str = "new",
    limit: int = 100,
    max_pages: int = 3,
) -> Iterator[dict]:
    after: Optional[str] = None
    for _ in range(max_pages):
        url = f"{BASE}/r/{subreddit}/{sort}.json?limit={limit}"
        if after:
            url += f"&after={after}"
        data = get_json(session, url)
        children = data.get("data", {}).get("children", [])
        if not children:
            break
        for c in children:
            yield c.get("data", {})
        after = data.get("data", {}).get("after")
        if not after:
            break
        sleep_jitter(1.5, 3.5)


def fetch_post_with_comments(
    session: requests.Session, post_id: str, comment_limit: int = 500
) -> dict:
    url = (
        f"https://www.reddit.com/comments/{post_id}.json?"
        f"limit={comment_limit}&sort=top"
    )
    return get_json(session, url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subreddit", required=True)
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--out", default="out")
    ap.add_argument("--sort", default="new", choices=["new", "hot", "top", "rising"])
    ap.add_argument("--no_comments", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    posts_path = out / f"{args.subreddit}_posts.ndjson"
    comments_path = out / f"{args.subreddit}_comments.ndjson"

    with requests.Session() as s:
        with posts_path.open("a", encoding="utf-8") as fp:
            if not args.no_comments:
                fc = comments_path.open("a", encoding="utf-8")
            else:
                fc = None

            try:
                for post in iter_sub_posts(
                    s, args.subreddit, sort=args.sort, limit=args.limit, max_pages=args.pages
                ):
                    fp.write(json.dumps(post, ensure_ascii=False) + "\n")
                    fp.flush()
                    sleep_jitter(1.0, 2.0)

                    if fc is not None:
                        post_id = post.get("id")
                        if not post_id:
                            continue
                        try:
                            payload = fetch_post_with_comments(s, post_id, comment_limit=500)
                            fc.write(
                                json.dumps(
                                    {"post_id": post_id, "payload": payload}, ensure_ascii=False
                                )
                                + "\n"
                            )
                            fc.flush()
                        except Exception as e:
                            print("comment fetch failed", post_id, e)
                        sleep_jitter(1.5, 3.5)
            finally:
                if fc is not None:
                    fc.close()


if __name__ == "__main__":
    main()


