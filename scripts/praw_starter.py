"""
Polite PRAW starter script with rate limiting and optional proxy.
Usage:
  export REDDIT_CLIENT_ID=...
  export REDDIT_CLIENT_SECRET=...
  export REDDIT_USER_AGENT='app by u/yourname'
  python scripts/praw_starter.py --subreddit AskReddit --limit 10 --sort hot
"""

import os
import time
import argparse
import praw


def create_reddit():
    return praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT', 'scrapehive-starter/0.1'),
        username=os.getenv('REDDIT_USERNAME'),
        password=os.getenv('REDDIT_PASSWORD')
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subreddit', required=True)
    ap.add_argument('--limit', type=int, default=25)
    ap.add_argument('--sort', default='hot', choices=['hot', 'new', 'top', 'rising'])
    args = ap.parse_args()

    reddit = create_reddit()
    sub = reddit.subreddit(args.subreddit)
    if args.sort == 'hot':
        it = sub.hot(limit=args.limit)
    elif args.sort == 'new':
        it = sub.new(limit=args.limit)
    elif args.sort == 'top':
        it = sub.top(limit=args.limit)
    else:
        it = sub.rising(limit=args.limit)

    for i, s in enumerate(it, 1):
        print(f"{i:03d} [{s.score:>5}] {s.title} (comments: {s.num_comments}) -> https://reddit.com{s.permalink}")
        time.sleep(0.1)  # polite pacing


if __name__ == '__main__':
    main()


