"""
RedditAuthManager stub to create a PRAW client from environment variables.
Falls back to unauthenticated mode when credentials are missing.
"""

import os
import logging
from typing import Optional

try:
    import praw
except Exception:
    praw = None

class RedditAuthManager:
    def __init__(self):
        self.logger = logging.getLogger('reddit_auth_manager')
        self.accounts = bool(os.getenv('REDDIT_CLIENT_ID') and os.getenv('REDDIT_CLIENT_SECRET') and os.getenv('REDDIT_USER_AGENT'))
    
    def get_reddit_instance(self) -> Optional["praw.Reddit"]:
        if not praw:
            self.logger.warning('praw not installed; cannot create Reddit API client')
            return None
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT', 'scrapehive/0.1')
        username = os.getenv('REDDIT_USERNAME')
        password = os.getenv('REDDIT_PASSWORD')
        if not client_id or not client_secret:
            self.logger.info('No Reddit API credentials found; using unauthenticated mode')
            return None
        try:
            if username and password:
                return praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent,
                    username=username,
                    password=password
                )
            else:
                return praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
        except Exception as e:
            self.logger.error(f'Failed to initialize Reddit client: {e}')
            return None

