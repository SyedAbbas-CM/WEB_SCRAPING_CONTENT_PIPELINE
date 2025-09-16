"""
Minimal ProxyManager stub that wraps proxy providers and returns usable proxies.
Extend with rotation, scoring, and provider selection as needed.
"""

import random
from typing import Dict, List, Optional

class ProxyManager:
    def __init__(self):
        self._active: List[Dict] = []
        # Simple in-memory pool; integrate with utils.proxy_providers for real sources
        self._pool: List[Dict] = []
        self._scores: Dict[str, float] = {}
    
    def add_proxy(self, proxy_url: str, platform: Optional[str] = None):
        self._pool.append({'url': proxy_url, 'platform': platform})
        self._scores[proxy_url] = self._scores.get(proxy_url, 1.0)
    
    def get_proxy(self, platform: Optional[str] = None, exclude: Optional[List[Dict]] = None) -> Optional[Dict]:
        exclude_urls = {p['url'] for p in (exclude or [])}
        candidates = [p for p in self._pool if p['url'] not in exclude_urls and (not platform or p.get('platform') in (None, platform))]
        if not candidates:
            return None
        # Choose by weighted score
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda p: self._scores.get(p['url'], 1.0), reverse=True)
        proxy = candidates[0]
        self._active.append(proxy)
        return proxy
    
    def mark_bad_proxy(self, proxy: Optional[Dict]):
        if not proxy:
            return
        # Lower score and potentially remove
        url = proxy.get('url')
        if not url:
            return
        self._scores[url] = max(0.1, self._scores.get(url, 1.0) * 0.5)
        # Optionally drop if too low
        if self._scores[url] < 0.2:
            self._pool = [p for p in self._pool if p['url'] != url]
            self._active = [p for p in self._active if p['url'] != url]
    
    def get_active_count(self) -> int:
        return len(self._active)

    def mark_good_proxy(self, proxy: Optional[Dict]):
        if not proxy:
            return
        url = proxy.get('url')
        if not url:
            return
        self._scores[url] = min(5.0, self._scores.get(url, 1.0) * 1.2)

