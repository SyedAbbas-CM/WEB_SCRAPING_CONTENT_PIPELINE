"""
Advanced ProxyManager with health scoring, stickiness, cooldowns, and block handling.
Backwards-compatible public API for existing nodes/scrapers.
"""

import time
import random
from typing import Dict, List, Optional, Tuple


class ProxyManager:
    def __init__(self):
        # Pool of proxies as dicts: {'url': str, 'platform': Optional[str]}
        self._pool: List[Dict] = []
        # Active allocations (best-effort tracking for metrics)
        self._active: List[Dict] = []
        # Per-proxy metadata keyed by URL
        self._meta: Dict[str, Dict] = {}

    # -------------------------
    # Pool management
    # -------------------------
    def add_proxy(self, proxy_url: str, platform: Optional[str] = None):
        if not proxy_url:
            return
        if any(p.get('url') == proxy_url for p in self._pool):
            # Already present; lightly bump score
            m = self._meta.get(proxy_url)
            if m:
                m['score'] = min(5.0, m['score'] * 1.05)
            return
        proxy = {'url': proxy_url, 'platform': platform}
        self._pool.append(proxy)
        self._meta[proxy_url] = {
            'score': 1.0,
            'last_used': 0.0,
            'successes': 0,
            'failures': 0,
            'consecutive_failures': 0,
            'cooldown_until': 0.0,
            'sticky_until': 0.0,
            'blocked_sites': set(),
            'last_status': None,
        }

    # -------------------------
    # Selection
    # -------------------------
    def get_proxy(self, platform: Optional[str] = None, exclude: Optional[List[Dict]] = None,
                  sticky_minutes: int = 5) -> Optional[Dict]:
        now = time.time()
        exclude_urls = {p['url'] for p in (exclude or []) if p and p.get('url')}

        def candidate_filter(p: Dict) -> bool:
            url = p.get('url')
            m = self._meta.get(url, {})
            if not url or url in exclude_urls:
                return False
            if platform and p.get('platform') not in (None, platform):
                return False
            if m.get('cooldown_until', 0) > now:
                return False
            if platform and platform in m.get('blocked_sites', set()):
                return False
            return True

        candidates = [p for p in self._pool if candidate_filter(p)]
        if not candidates:
            return None

        # Rank by score, then least recently used
        def rank_key(p: Dict) -> Tuple[float, float]:
            m = self._meta.get(p['url'], {})
            return (m.get('score', 1.0), -m.get('last_used', 0.0))

        candidates.sort(key=rank_key, reverse=True)
        chosen = candidates[0]

        # Mark stickiness and usage
        meta = self._meta.get(chosen['url'])
        meta['last_used'] = now
        if sticky_minutes and sticky_minutes > 0:
            meta['sticky_until'] = max(meta.get('sticky_until', 0.0), now + sticky_minutes * 60.0)

        self._active.append(chosen)
        return chosen

    # -------------------------
    # Feedback APIs
    # -------------------------
    def mark_good_proxy(self, proxy: Optional[Dict]):
        if not proxy:
            return
        url = proxy.get('url')
        if not url or url not in self._meta:
            return
        m = self._meta[url]
        m['score'] = min(5.0, m.get('score', 1.0) * 1.2)
        m['successes'] = m.get('successes', 0) + 1
        m['consecutive_failures'] = 0

    def mark_bad_proxy(self, proxy: Optional[Dict], reason: Optional[str] = None, platform: Optional[str] = None):
        if not proxy:
            return
        url = proxy.get('url')
        if not url or url not in self._meta:
            return
        m = self._meta[url]
        m['failures'] = m.get('failures', 0) + 1
        m['consecutive_failures'] = m.get('consecutive_failures', 0) + 1
        # Decrease score (more if consecutive failures are high)
        decay = 0.5 if m['consecutive_failures'] <= 3 else 0.3
        m['score'] = max(0.1, m.get('score', 1.0) * decay)

        # Cooldown with exponential backoff and jitter
        base = 60  # 1 minute base
        backoff = min(base * (2 ** (m['consecutive_failures'] - 1)), 3600)
        jitter = random.randint(0, 30)
        m['cooldown_until'] = time.time() + backoff + jitter

        # Optionally flag platform as blocked after repeated failures
        if platform and m['consecutive_failures'] >= 3:
            m.setdefault('blocked_sites', set()).add(platform)

        # Drop from pool if score is too low
        if m['score'] < 0.2 and m['failures'] >= 5:
            self._pool = [p for p in self._pool if p.get('url') != url]
            self._active = [p for p in self._active if p.get('url') != url]
            self._meta.pop(url, None)

    def record_result(self, proxy: Optional[Dict], success: bool, status_code: Optional[int] = None,
                      platform: Optional[str] = None):
        """Convenience helper to update health based on request outcome."""
        if not proxy:
            return
        url = proxy.get('url')
        if not url or url not in self._meta:
            return
        m = self._meta[url]
        m['last_status'] = status_code
        if success:
            self.mark_good_proxy(proxy)
            return

        # Interpret typical block/rate-limit signals
        if status_code in (403, 401):
            self.mark_bad_proxy(proxy, reason='forbidden', platform=platform)
        elif status_code == 429:
            # Rate limited: cooldown but do not permanently punish as harshly
            m['consecutive_failures'] = m.get('consecutive_failures', 0) + 1
            m['score'] = max(0.2, m.get('score', 1.0) * 0.7)
            retry_cooldown = 300 + random.randint(0, 60)  # ~5-6 minutes
            m['cooldown_until'] = time.time() + retry_cooldown
        elif status_code and status_code >= 500:
            # Server errors: brief cooldown
            m['score'] = max(0.2, m.get('score', 1.0) * 0.8)
            m['cooldown_until'] = time.time() + 120 + random.randint(0, 30)
        else:
            self.mark_bad_proxy(proxy, reason='error', platform=platform)

    # -------------------------
    # Metrics
    # -------------------------
    def get_active_count(self) -> int:
        return len(self._active)

    def get_stats(self) -> Dict:
        total = len(self._pool)
        blocked = 0
        cooling = 0
        now = time.time()
        for p in self._pool:
            m = self._meta.get(p['url'], {})
            if m.get('cooldown_until', 0) > now:
                cooling += 1
            if m.get('blocked_sites'):
                blocked += 1
        return {
            'total': total,
            'active': len(self._active),
            'cooling_down': cooling,
            'blocked_flagged': blocked,
            'pool': [
                {
                    'url': p['url'],
                    'score': round(self._meta.get(p['url'], {}).get('score', 1.0), 2),
                    'cooldown_until': self._meta.get(p['url'], {}).get('cooldown_until', 0),
                    'sticky_until': self._meta.get(p['url'], {}).get('sticky_until', 0),
                    'successes': self._meta.get(p['url'], {}).get('successes', 0),
                    'failures': self._meta.get(p['url'], {}).get('failures', 0),
                }
                for p in self._pool
            ]
        }

