"""
AntiDetection v2: header ordering, UA rotation, randomized delays, httpx client helpers.
TLS/JA3 and true HTTP2 fingerprinting left as optional to integrate with tls-client later.
"""

import os
import random
import time
from typing import Dict, Optional

try:
    import httpx
except Exception:
    httpx = None

from .anti_detection import AntiDetection


class AntiDetectionV2(AntiDetection):
    def __init__(self):
        super().__init__()
        # User-agent pools (pinned)
        self.chrome_ua_list = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        ]

    def get_pinned_ua(self) -> str:
        return random.choice(self.chrome_ua_list)

    def build_headers(self, platform: str, minimal: bool = False) -> Dict:
        if minimal:
            return {
                'User-Agent': self.get_pinned_ua(),
                'Accept': 'application/json,text/html;q=0.8,*/*;q=0.5',
                'Accept-Language': 'en-US,en;q=0.9',
                'Connection': 'keep-alive'
            }
        return dict(self.get_headers(platform))

    def cautious_sleep(self, base: float = 0.5, jitter: float = 0.75):
        time.sleep(base + random.random() * jitter)

    def build_httpx_client(self, proxy_url: Optional[str] = None, http2: bool = True) -> Optional["httpx.Client"]:
        if not httpx:
            return None
        proxies = None
        if proxy_url:
            proxies = {
                'http://': proxy_url,
                'https://': proxy_url
            }
        return httpx.Client(http2=http2, proxies=proxies, timeout=30.0)

